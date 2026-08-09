#!/usr/bin/env python3
"""Six-servo DYNAMIXEL bus for the tendon-driven continuum arm.

Wraps the six XL330-M288-T servos behind one object so the rest of the stack
never touches the SDK. SyncWrite/SyncRead throughout: all six goal positions
leave in a single packet, so the two arm sections actuate together instead of
in sequence.

This is the servo half of `real_arm.py`. When the camera is wired in, the
missing piece is `forward(q)` calling `move_and_settle()` and then reading
marker poses instead of returning servo positions.

Standalone self-test (safe with no cables attached):
    python servo_bus.py --selftest

Capture the home pose once the tendons are strung just-taut:
    python servo_bus.py --set-home

    pip install dynamixel-sdk
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
    COMM_SUCCESS, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD,
)

# --------------------------------------------------------------------------- #
# CONFIGURE THESE THREE THINGS FIRST
# --------------------------------------------------------------------------- #
PORT = "/dev/tty.usbserial-FTBENWAP"   # macOS: ls /dev/tty.usbserial-*
BAUDRATE = 57600                        # must match what you set in the Wizard
IDS = [1, 2, 3, 4, 5, 6]

HOME_FILE = "servo_home.json"

# --------------------------------------------------------------------------- #
# X-series control table (protocol 2.0). Cross-check against the XL330-M288
# e-manual if anything behaves oddly - addresses differ between series.
# --------------------------------------------------------------------------- #
ADDR_RETURN_DELAY     = 9     # 1 byte, EEPROM, unit 2us (default 250 = 500us)
ADDR_DRIVE_MODE       = 10    # 1 byte, EEPROM  (bit0 reverse, bit2 torque-on-goal)
ADDR_OPERATING_MODE   = 11    # 1 byte, EEPROM  (torque must be OFF to write)
ADDR_HOMING_OFFSET    = 20    # 4 byte, EEPROM  (silently shifts reported position)
ADDR_MAX_POS_LIMIT    = 48    # 4 byte, EEPROM  (clamps silently in mode 3)
ADDR_MIN_POS_LIMIT    = 52    # 4 byte, EEPROM
ADDR_SHUTDOWN         = 63    # 1 byte, EEPROM  (which faults auto-disable torque)
ADDR_STATUS_LEVEL     = 68    # 1 byte  (2 = reply to everything; <2 hides write acks)
ADDR_BUS_WATCHDOG     = 98    # 1 byte  (non-zero = stop if the bus goes quiet)
ADDR_CURRENT_LIMIT    = 38    # 2 byte, EEPROM
ADDR_TORQUE_ENABLE    = 64    # 1 byte
ADDR_LED              = 65    # 1 byte
ADDR_HW_ERROR_STATUS  = 70    # 1 byte
ADDR_PROFILE_ACCEL    = 108   # 4 byte
ADDR_PROFILE_VELOCITY = 112   # 4 byte
ADDR_GOAL_POSITION    = 116   # 4 byte
ADDR_MOVING           = 122   # 1 byte
ADDR_PRESENT_CURRENT  = 126   # 2 byte
ADDR_PRESENT_POSITION = 132   # 4 byte

LEN_GOAL_POSITION     = 4
LEN_PRESENT_POSITION  = 4

MODE_POSITION          = 3    # single turn, 0..4095
MODE_EXTENDED_POSITION = 4    # multi-turn - what the capstans need
MODE_CURRENT_POSITION  = 5    # position with a force ceiling

UNITS_PER_REV = 4096          # 0.088 deg per unit

# Deliberately slow defaults. A servo at full speed with a tendon attached can
# snap cable or split a disc before you react.
SAFE_PROFILE_VELOCITY = 60    # 0 = unlimited. 60 is a gentle crawl.
SAFE_PROFILE_ACCEL    = 20
SAFE_CURRENT_LIMIT    = 400   # ~40% of the 1.47 A stall. Raise once calibrated.

# Software travel envelope, in revolutions either side of home. Tightened once
# you have measured real tendon travel.
MAX_REV_FROM_HOME = 2.0

# MEASURED, not guessed. SyncWrite is a broadcast instruction: the servos send
# no status packet, so the SDK returns the instant the bytes are queued. If the
# next packet goes out before the servos have finished taking the write, they
# discard it - silently, with COMM_SUCCESS reported and no error flag set.
#
# Sweep on this rig at 57600 baud, six servos, measuring how many acted on the
# command as a function of the gap before the following SyncRead:
#       0 ms -> 0/6      5 ms -> 0/6      10 ms -> 6/6      20 ms -> 6/6
# The cliff sits just under 10 ms, so 25 ms is roughly 2.5x margin. Raise it if
# you ever add servos or drop the baud rate.
POST_SYNCWRITE_DELAY = 0.025


def _signed32(v: int) -> int:
    """SDK returns unsigned; extended-position values can be negative."""
    return v - (1 << 32) if v > 0x7FFFFFFF else v


class ServoBusError(RuntimeError):
    pass


class ServoBus:
    def __init__(self, port: str = PORT, baud: int = BAUDRATE, ids=None):
        self.ids = list(ids or IDS)
        self.port = PortHandler(port)
        self.packet = PacketHandler(2.0)
        self._port_name, self._baud = port, baud
        self.home = {}          # id -> raw position defined as "arm straight"
        self._torque_on = False

    # ---------------- lifecycle ---------------- #
    def connect(self):
        if not self.port.openPort():
            raise ServoBusError(
                f"could not open {self._port_name}\n"
                "  - is the U2D2 plugged in?   ls /dev/tty.usbserial-*\n"
                "  - is another program (DYNAMIXEL Wizard) holding the port?"
            )
        if not self.port.setBaudRate(self._baud):
            raise ServoBusError(f"could not set baud {self._baud}")
        missing = [i for i in self.ids if not self.ping(i)]
        if missing:
            raise ServoBusError(
                f"no response from IDs {missing}\n"
                "  - servos powered? the U2D2 supplies signal only\n"
                "  - 3-pin TTL port, not the 4-pin RS-485 one\n"
                f"  - do all six actually sit at baud {self._baud}?"
            )
        self._sync_write = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        self._sync_read = GroupSyncRead(
            self.port, self.packet, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        for i in self.ids:
            self._sync_read.addParam(i)
        self.load_home()
        return self

    def close(self):
        try:
            if self._torque_on:
                self.torque(False)
        finally:
            self.port.closePort()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---------------- low level ---------------- #
    def _check(self, res, err, what, dxl_id):
        if res != COMM_SUCCESS:
            raise ServoBusError(f"{what} id={dxl_id}: {self.packet.getTxRxResult(res)}")
        if err:
            raise ServoBusError(f"{what} id={dxl_id}: {self.packet.getRxPacketError(err)}")

    def ping(self, dxl_id) -> bool:
        _, res, err = self.packet.ping(self.port, dxl_id)
        return res == COMM_SUCCESS and err == 0

    def _w1(self, dxl_id, addr, val):
        res, err = self.packet.write1ByteTxRx(self.port, dxl_id, addr, val)
        self._check(res, err, f"write1@{addr}", dxl_id)

    def _w2(self, dxl_id, addr, val):
        res, err = self.packet.write2ByteTxRx(self.port, dxl_id, addr, val)
        self._check(res, err, f"write2@{addr}", dxl_id)

    def _w4(self, dxl_id, addr, val):
        res, err = self.packet.write4ByteTxRx(self.port, dxl_id, addr, val)
        self._check(res, err, f"write4@{addr}", dxl_id)

    def _r1(self, dxl_id, addr):
        v, res, err = self.packet.read1ByteTxRx(self.port, dxl_id, addr)
        self._check(res, err, f"read1@{addr}", dxl_id)
        return v

    def _r4(self, dxl_id, addr):
        v, res, err = self.packet.read4ByteTxRx(self.port, dxl_id, addr)
        self._check(res, err, f"read4@{addr}", dxl_id)
        return _signed32(v)

    # ---------------- configuration ---------------- #
    def torque(self, on: bool):
        for i in self.ids:
            self._w1(i, ADDR_TORQUE_ENABLE, 1 if on else 0)
        self._torque_on = on

    def configure(self, mode: int = MODE_EXTENDED_POSITION):
        """Torque OFF -> write EEPROM -> torque stays off. Call before enabling.

        Operating mode and current limit live in EEPROM and are silently
        ignored while torque is enabled - the single most common reason a mode
        change 'does not take'.
        """
        self.torque(False)
        for i in self.ids:
            self._w1(i, ADDR_OPERATING_MODE, mode)
            self._w2(i, ADDR_CURRENT_LIMIT, SAFE_CURRENT_LIMIT)
        for i in self.ids:                       # RAM, order does not matter
            self._w4(i, ADDR_PROFILE_VELOCITY, SAFE_PROFILE_VELOCITY)
            self._w4(i, ADDR_PROFILE_ACCEL, SAFE_PROFILE_ACCEL)

    def led(self, dxl_id, on: bool):
        self._w1(dxl_id, ADDR_LED, 1 if on else 0)

    def hardware_errors(self) -> dict:
        return {i: self._r1(i, ADDR_HW_ERROR_STATUS) for i in self.ids}

    def audit_config(self) -> bool:
        """Dump the registers that fail *silently* when set wrong.

        Everything here reports COMM_SUCCESS regardless, so a bad value shows
        up as "the servo ignored me" rather than as an error. Measured on this
        rig, all six sit at factory defaults; this exists so that if one drifts
        you find out in seconds instead of an afternoon.
        """
        expect = {
            "status_level":  (ADDR_STATUS_LEVEL, 1, 2,
                              "<2 stops the servo acking writes; writes look dead"),
            "bus_watchdog":  (ADDR_BUS_WATCHDOG, 1, 0,
                              "non-zero halts motion when the bus goes quiet"),
            "drive_mode":    (ADDR_DRIVE_MODE, 1, 0,
                              "bit0 reverses direction - would invert a tendon"),
            "homing_offset": (ADDR_HOMING_OFFSET, 4, 0,
                              "silently shifts every position you read"),
        }
        print(f"{'id':>3} " + " ".join(f"{k:>14}" for k in expect) + "   ret_delay")
        clean = True
        for i in self.ids:
            row, bad = [], []
            for name, (addr, size, want, _) in expect.items():
                v = self._r1(i, addr) if size == 1 else self._r4(i, addr)
                row.append(f"{v:>14}")
                if v != want:
                    bad.append(name)
                    clean = False
            rdel = self._r1(i, ADDR_RETURN_DELAY)
            print(f"{i:>3} " + " ".join(row) + f"   {rdel} ({rdel*2}us)"
                  + (f"   <-- CHECK {bad}" if bad else ""))
        if clean:
            print("all servos at expected values")
        else:
            for name, (_, _, want, why) in expect.items():
                print(f"  {name}: expected {want} - {why}")
        return clean

    # ---------------- motion ---------------- #
    def read_positions(self, retries: int = 3) -> dict:
        """SyncRead all six, retrying transient bus errors.

        A dropped status packet while the servos are moving is usually a
        momentary bus or supply hiccup rather than a real fault, so a couple
        of retries turn a hard failure into a hiccup. Persistent failure gets
        diagnosed below rather than reported as a bare comms error.
        """
        last = ""
        for attempt in range(retries):
            res = self._sync_read.txRxPacket()
            if res == COMM_SUCCESS:
                out, missing = {}, []
                for i in self.ids:
                    if self._sync_read.isAvailable(i, ADDR_PRESENT_POSITION,
                                                   LEN_PRESENT_POSITION):
                        out[i] = _signed32(self._sync_read.getData(
                            i, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION))
                    else:
                        missing.append(i)
                if not missing:
                    return out
                last = f"no data from ids {missing}"
            else:
                last = self.packet.getTxRxResult(res).strip()
            time.sleep(0.05 * (attempt + 1))
        raise ServoBusError(self._diagnose(last))

    def _diagnose(self, symptom: str) -> str:
        """Distinguish a browned-out servo from a bus glitch by re-pinging."""
        alive = {i: self.ping(i) for i in self.ids}
        dead = [i for i, ok in alive.items() if not ok]
        msg = [f"sync read failed after retries: {symptom}"]
        if dead:
            msg += [
                f"  ids {dead} no longer respond to ping -> they lost power or reset.",
                "  Almost always an undersized supply: six XL330s starting together",
                "  draw a current spike, the rail sags, and servos brown out. This",
                "  looks exactly like a comms bug but is not one.",
                "  Check the adapter is 5 V and rated >= 5 A.",
            ]
        else:
            msg += [
                "  all ids still ping -> the servos are alive, so this is a bus",
                "  timing problem rather than a power one.",
                "  Try a faster baud (--baud 1000000) so each SyncRead spends less",
                "  time on the wire, or a longer --poll interval.",
            ]
        return "\n".join(msg)

    def write_positions(self, targets: dict):
        """One packet, all six servos. Targets clipped to the travel envelope."""
        self._sync_write.clearParam()
        for i, raw in targets.items():
            raw = int(self._clip(i, raw))
            ok = self._sync_write.addParam(i, bytes([
                DXL_LOBYTE(DXL_LOWORD(raw)), DXL_HIBYTE(DXL_LOWORD(raw)),
                DXL_LOBYTE(DXL_HIWORD(raw)), DXL_HIBYTE(DXL_HIWORD(raw)),
            ]))
            if not ok:
                raise ServoBusError(
                    f"SyncWrite addParam rejected id {i} -- the packet would "
                    "have been sent without this servo, which looks like the "
                    "servo silently ignoring a command.")
        res = self._sync_write.txPacket()
        if res != COMM_SUCCESS:
            raise ServoBusError(f"sync write: {self.packet.getTxRxResult(res)}")
        # Do not remove: without this the very next packet silently voids the
        # write. See POST_SYNCWRITE_DELAY for the measurement behind it.
        time.sleep(POST_SYNCWRITE_DELAY)

    def _clip(self, dxl_id, raw):
        if dxl_id not in self.home:
            return raw
        span = MAX_REV_FROM_HOME * UNITS_PER_REV
        lo, hi = self.home[dxl_id] - span, self.home[dxl_id] + span
        return max(lo, min(hi, raw))

    def move_one_at_a_time(self, targets: dict, settle: float = 0.4) -> dict:
        """Diagnostic: move servos sequentially so only one draws current.

        If the six-at-once move browns out but this succeeds, the supply
        cannot handle the simultaneous inrush and the fix is a bigger supply,
        not different code.
        """
        for i, raw in targets.items():
            self.write_positions({i: raw})
            time.sleep(settle)
        return self.read_positions()

    def move_and_settle(self, targets: dict, settle: float = 1.0,
                        tol: int = 10, timeout: float = 5.0) -> dict:
        """Command, wait until motion stops (or timeout), then hold `settle`.

        The fixed settle is deliberate: it makes every measurement a snapshot
        taken under an identical protocol, matching how the simulated target
        domain was sampled.
        """
        self.write_positions(targets)
        t0 = time.time()
        while time.time() - t0 < timeout:
            pos = self.read_positions()
            if all(abs(pos[i] - self._clip(i, targets[i])) <= tol for i in targets):
                break
            time.sleep(0.05)   # 57600 baud needs breathing room between SyncReads
        time.sleep(settle)
        return self.read_positions()

    # ---------------- home pose ---------------- #
    def set_home(self):
        """Call with tendons strung just-taut and the arm straight."""
        self.home = self.read_positions()
        with open(HOME_FILE, "w") as f:
            json.dump({str(k): v for k, v in self.home.items()}, f, indent=2)
        print(f"home written to {HOME_FILE}: {self.home}")

    def load_home(self):
        if os.path.exists(HOME_FILE):
            self.home = {int(k): v for k, v in json.load(open(HOME_FILE)).items()}

    def go_home(self, settle: float = 1.0):
        if not self.home:
            raise ServoBusError(f"no {HOME_FILE}; run --set-home first")
        return self.move_and_settle(dict(self.home), settle=settle)

    # ---------------- the interface real_arm.py will use ---------------- #
    def q_to_positions(self, q, travel_rev: float = 1.0) -> dict:
        """Normalized actuation q in [0,1]^6 -> raw servo positions.

        q=0 is home (tendon slack limit), q=1 is `travel_rev` revolutions of
        take-up. Replace travel_rev with the per-servo value measured during
        spool characterisation.
        """
        if not self.home:
            raise ServoBusError(f"no {HOME_FILE}; run --set-home first")
        if len(q) != len(self.ids):
            raise ValueError(f"expected {len(self.ids)} values, got {len(q)}")
        return {i: int(self.home[i] + float(qi) * travel_rev * UNITS_PER_REV)
                for i, qi in zip(self.ids, q)}


# --------------------------------------------------------------------------- #
def selftest(bus: ServoBus, delta_rev: float = 0.1, sequential: bool = False):
    print(f"connected on {bus._port_name} @ {bus._baud}")
    print("ping        :", {i: bus.ping(i) for i in bus.ids})
    print("hw errors   :", bus.hardware_errors(), "(all 0 is good)")

    print("\nidentifying servos by LED, one at a time...")
    for i in bus.ids:
        bus.led(i, True)
        print(f"  id {i} lit", end="\r", flush=True)
        time.sleep(0.4)
        bus.led(i, False)
    print("  LED sweep done       ")

    bus.configure(MODE_EXTENDED_POSITION)
    print("\nmode=extended position, profile velocity capped, current limited")

    bus.torque(True)
    start = bus.read_positions()
    print("start       :", start)

    delta = int(delta_rev * UNITS_PER_REV)
    targets = {i: p + delta for i, p in start.items()}

    if sequential:
        print(f"\ncommanding +{delta_rev} rev SEQUENTIALLY (one servo at a time)")
        reached = bus.move_one_at_a_time(targets)
    else:
        print(f"\ncommanding +{delta_rev} rev on all six (SyncWrite, one packet)")
        reached = bus.move_and_settle(targets, settle=0.5)

    print(f"\n{'id':>3} {'target':>9} {'reached':>9} {'error':>7}")
    worst = 0
    for i in bus.ids:
        err = reached[i] - targets[i]
        worst = max(worst, abs(err))
        print(f"{i:>3} {targets[i]:>9} {reached[i]:>9} {err:>7}")
    print(f"\nworst error {worst} units = {worst * 360 / UNITS_PER_REV:.2f} deg")

    print("returning to start")
    bus.move_and_settle(start, settle=0.3)
    bus.torque(False)
    print("torque off. self-test complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--baud", type=int, default=BAUDRATE)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--set-home", action="store_true")
    ap.add_argument("--go-home", action="store_true")
    ap.add_argument("--delta-rev", type=float, default=0.1,
                    help="self-test move size in revolutions")
    ap.add_argument("--sequential", action="store_true",
                    help="move one servo at a time; isolates power problems "
                         "from bus-timing problems")
    ap.add_argument("--audit", action="store_true",
                    help="dump the registers that fail silently when wrong")
    args = ap.parse_args()

    bus = ServoBus(args.port, args.baud)
    try:
        bus.connect()
        if args.audit:
            bus.audit_config()
        elif args.set_home:
            bus.torque(False)          # back-drivable so you can pose it by hand
            input("pose the arm straight with tendons just taut, then Enter...")
            bus.set_home()
        elif args.go_home:
            bus.configure(MODE_EXTENDED_POSITION)
            bus.torque(True)
            print("at home:", bus.go_home())
        else:
            selftest(bus, args.delta_rev, args.sequential)
    except ServoBusError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted - releasing torque")
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
