"""Tests for motion.py: calibration, lane crossing, jump/crouch latches,
running cadence and the pause/resume state machine.

Every scenario drives MotionDetector/ControllerStateMachine with synthetic
pose frames at 30 Hz (dt = 33 ms).  EWMA settling shifts the exact frame an
event fires, so assertions count events and check windows, not single frames.
"""

from __future__ import annotations

import pytest

from game2health.motion import (
    CalibrationPhase,
    CalibrationResult,
    CalibrationTracker,
    ControllerStateMachine,
    CROUCH_KNEE_MAX_DEG,
    CROUCH_REARM_KNEE_MIN_DEG,
    MotionDetector,
    _knee_angle_raw,
)
from game2health.types import (
    Action,
    ControllerMode,
    Lane,
    MotionSnapshot,
)

from helpers import (
    BEND_OVER,
    CROUCH_DOWN,
    CROUCH_LIGHT,
    JUMP_UP,
    RUN_LIFT_LEFT,
    RUN_LIFT_RIGHT,
    PoseBuilder,
    run_motion,
)

W_43, H_43 = 800, 600
W_169, H_169 = 1280, 720

# Baseline matching the synthetic standing pose (torso 0.20).
CAL = CalibrationResult(
    hip_y_left=0.56, hip_y_right=0.56,
    knee_y_left=0.74, knee_y_right=0.74,
    ankle_y_left=0.93, ankle_y_right=0.93,
    hip_mid_y=0.56, torso_length=0.20, upright_knee_angle_deg=178.0,
)


def make_detector() -> MotionDetector:
    det = MotionDetector()
    det.set_baseline(CAL)
    return det


def seg_stand(center_raw: float = 0.5) -> dict:
    return dict(center_raw=center_raw)


def jog_segments(n_cycles: int, center_raw: float = 0.5):
    """Alternating L/R single-leg lifts with ground contact between them.

    Each cycle: lift one leg for 6 frames (~200 ms), then back on the
    ground for 6 frames so the leg re-arms below the re-arm threshold.
    """
    segs = []
    for i in range(n_cycles):
        lift = RUN_LIFT_LEFT if i % 2 == 0 else RUN_LIFT_RIGHT
        segs.append((6, dict(center_raw=center_raw, deltas=lift)))
        segs.append((6, dict(center_raw=center_raw)))
    return segs


def count_actions(actions, action) -> int:
    return sum(1 for a in actions if a is action or a == action)


def run_active(builder, jog_cycles: int = 8):
    """Stand CENTER, jog to ACTIVE.  Returns (detector, sm, transcript)."""
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    transcript = run_motion(det, sm, builder.stream(
        [(12, seg_stand()), *jog_segments(jog_cycles)],
    ))
    assert sm.mode is ControllerMode.ACTIVE
    return det, sm, transcript


# ---------------------------------------------------------------------------
# Aspect consistency
# ---------------------------------------------------------------------------



def _after(transcript):
    """Next monotonic timestamp after a transcript (continuation runs)."""
    return transcript[-1][0] + 33


def test_knee_angle_identical_across_aspects():
    b43 = PoseBuilder(W_43, H_43)
    b169 = PoseBuilder(W_169, H_169)
    for side in ((23, 25, 27), (24, 26, 28)):
        for deltas in (CROUCH_DOWN, None):
            s43 = b43.sample(0, deltas=deltas)
            s169 = b169.sample(0, deltas=deltas)
            a43 = _knee_angle_raw(*[s43.landmarks[i] for i in side], s43)
            a169 = _knee_angle_raw(*[s169.landmarks[i] for i in side], s169)
            assert a43 == pytest.approx(a169, abs=1e-9)
            if deltas is CROUCH_DOWN:
                assert a43 < CROUCH_KNEE_MAX_DEG
            else:
                assert a43 > CROUCH_REARM_KNEE_MIN_DEG


def test_crouch_verdict_consistent_across_aspects():
    results = []
    for w, h in ((W_43, H_43), (W_169, H_169)):
        b = PoseBuilder(w, h)
        det = make_detector()
        events = []
        for _ts, sample in b.stream([
            (12, seg_stand()),
            (24, dict(deltas=CROUCH_DOWN)),
            (12, seg_stand()),
            (24, dict(deltas=CROUCH_DOWN)),
            (12, seg_stand()),
        ]):
            events.extend(det.update(sample).events)
        results.append(events)
    assert results[0] == results[1] == [Action.CROUCH, Action.CROUCH]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_gate_visibility_only():
    b = PoseBuilder(W_169, H_169)
    tr = CalibrationTracker()
    status = None
    for _ts, sample in b.stream([(200, dict(hidden=(15,)))]):
        status = tr.update(sample)
    assert status.phase is CalibrationPhase.FULL_BODY
    assert status.missing_indices == (15,)

    tr.reset()
    # Feet near the bottom edge (outside the old safe box) are accepted:
    # framing is no longer enforced, only visibility.
    deltas = {31: (0.0, 0.025), 32: (0.0, 0.025)}  # foot y 0.96 -> 0.985
    for _ts, sample in b.stream([(10, dict(deltas=deltas))]):
        status = tr.update(sample)
    assert status.phase is CalibrationPhase.FULL_BODY
    assert status.missing_indices == ()

def test_calibration_gap_resets_continuous_window():
    b = PoseBuilder(W_169, H_169)
    tr = CalibrationTracker()
    statuses = []
    for _ts, sample in b.stream([
        (30, seg_stand()),          # ~1 s
        (16, dict(hidden=(27,))),   # ankle missing ~0.5 s
        (30, seg_stand()),          # ~1 s more -> never 1.5 s continuous
    ]):
        statuses.append(tr.update(sample))
    assert statuses[-1].phase is CalibrationPhase.FULL_BODY
    assert statuses[-1].full_body_ms <= 1_000


def test_calibration_completes_with_valid_baseline():
    b = PoseBuilder(W_169, H_169)
    tr = CalibrationTracker()
    phase_seen = []
    status = None
    for _ts, sample in b.stream([(250, seg_stand())]):  # 8.25 s total
        status = tr.update(sample)
        phase_seen.append(status.phase)
    assert CalibrationPhase.UPRIGHT in phase_seen
    assert status.phase is CalibrationPhase.COMPLETE
    res = status.result
    assert res is not None and res.valid
    assert res.torso_length == pytest.approx(0.20, abs=0.01)
    assert res.hip_mid_y == pytest.approx(0.56, abs=0.01)
    assert res.ankle_y_left == pytest.approx(0.93, abs=0.01)
    assert res.upright_knee_angle_deg > 150.0


def test_calibration_never_completes_under_30_samples_per_window():
    b = PoseBuilder(W_169, H_169)
    tr = CalibrationTracker()
    # Samples every 100 ms: 2 s window collects ~20 < 30 required.
    ts = 0
    status = None
    for _ in range(60):  # 6 s
        status = tr.update(b.sample(ts))
        ts += 100
    assert status.result is None


# ---------------------------------------------------------------------------
# Lane crossings
# ---------------------------------------------------------------------------


def test_shoulder_lean_and_half_body_produce_no_lane_event():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    lean = {11: b.to_raw_x_delta(11, 0.15), 12: b.to_raw_x_delta(12, 0.15)}
    half = {11: b.to_raw_x_delta(11, 0.15), 23: b.to_raw_x_delta(23, 0.15),
            25: b.to_raw_x_delta(25, 0.15), 27: b.to_raw_x_delta(27, 0.15)}
    snaps = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (12, dict(deltas=lean)),
        (12, dict(deltas=half)),
        (12, seg_stand()),
    ]):
        snaps.append(det.update(sample))
    assert [e for s in snaps for e in s.events] == []
    assert snaps[-1].stable_lane is Lane.CENTER




def test_center_to_left_emits_single_move():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    snaps = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (20, dict(center_raw=0.15)),
    ]):
        snaps.append(det.update(sample))
    assert [e for s in snaps for e in s.events] == [Action.MOVE_LEFT]
    assert snaps[-1].stable_lane is Lane.LEFT


def test_center_to_right_fires_on_first_changed_frame_once():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    snaps = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (150, dict(center_raw=0.85)),
    ]):
        snaps.append(det.update(sample))
    fires = [i for i, snap in enumerate(snaps) if snap.events]
    assert fires == [12]
    assert snaps[12].events == (Action.MOVE_RIGHT,)

def test_nose_at_logged_right_edge_commits_right_lane():
    """The old hidden dead-zone rejected the observed x=0.656 position."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (20, dict(center_raw=0.656)),
    ]):
        events.extend(det.update(sample).events)
    assert events == [Action.MOVE_RIGHT]

def test_head_lane_survives_ui_ticks_and_low_ankle_visibility():
    """UI timeout ticks and leg confidence must not block head changes."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    events = []
    ts = 0
    for center in (0.50,) * 8 + (0.656,) * 8:
        sample = b.sample(ts, center_raw=center, hidden=(27, 28))
        events.extend(det.update(sample).events)
        events.extend(det.tick(ts + 16).events)
        ts += 33
    assert det.raw_lane is Lane.RIGHT
    assert det._stable_lane is Lane.RIGHT
    assert events == [Action.MOVE_RIGHT]


def test_left_to_right_sprint_fires_two_rights():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    snaps = []
    for _ts, sample in b.stream([
        (12, dict(center_raw=0.15)),
        (2, dict(center_raw=0.5)),
        (12, dict(center_raw=0.85)),
    ]):
        snaps.append(det.update(sample))
    events = [event for snap in snaps for event in snap.events]
    assert events == [
        Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_RIGHT,
    ]
    assert snaps[-1].stable_lane is Lane.RIGHT

def test_lane_uses_head_position_when_wide_body_spans_boundaries():
    """A zoomed/cropped body can be wider than one lane."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    pairs = ((11, 12), (23, 24), (25, 26), (27, 28))

    def wide_at(center):
        deltas = {0: b.to_raw_x_delta(0, center)}
        for left, right in pairs:
            deltas[left] = b.to_raw_x_delta(left, center - 0.17)
            deltas[right] = b.to_raw_x_delta(right, center + 0.17)
        return {"deltas": deltas}

    snaps = []
    for _ts, sample in b.stream([
        (12, wide_at(0.50)),
        (20, wide_at(0.18)),
    ]):
        snaps.append(det.update(sample))

    assert [e for s in snaps for e in s.events] == [Action.MOVE_LEFT]
    assert snaps[-1].stable_lane is Lane.LEFT


# ---------------------------------------------------------------------------
# Jump
# ---------------------------------------------------------------------------


def test_vertical_jump_fires_once_and_rearms():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    snaps = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (18, dict(deltas=JUMP_UP)),  # airborne ~600 ms
        (14, seg_stand()),           # land + re-arm
        (18, dict(deltas=JUMP_UP)),  # second jump
        (14, seg_stand()),
    ]):
        snaps.append(det.update(sample))
    fire_ts = [s.timestamp_ms for s in snaps if Action.JUMP in s.events]
    assert len(fire_ts) == 2
    assert fire_ts[1] - fire_ts[0] > 500  # no repeats while airborne
    assert snaps[-1].steps == 0  # two-foot takeoff is not a run step


def test_feet_up_hold_still_fires_jump():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    feet_up = {27: (0.0, -0.050), 28: (0.0, -0.050)}
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (18, dict(deltas=feet_up)),
        (14, seg_stand()),
        (18, dict(deltas=feet_up)),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.JUMP) == 2


def test_single_frame_feet_overlap_does_not_fire_jump():
    """A one-frame both-feet-up blink (jog crossover) is not a jump."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    feet_up = {27: (0.0, -0.050), 28: (0.0, -0.050)}
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (1, dict(deltas=feet_up)),
        (14, seg_stand()),
        (1, dict(deltas=feet_up)),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.JUMP) == 0


def test_both_ankles_below_higher_jump_line_do_not_fire():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    below_line = {27: (0.0, -0.030), 28: (0.0, -0.030)}
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (18, dict(deltas=below_line)),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.JUMP) == 0


def test_alternating_jog_does_not_fire_jump():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        *jog_segments(10),
        (12, seg_stand()),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.JUMP) == 0


# ---------------------------------------------------------------------------
# Crouch
# ---------------------------------------------------------------------------


def test_bend_over_with_straight_knees_does_not_crouch():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (30, dict(deltas=BEND_OVER)),
        (12, seg_stand()),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.CROUCH) == 0


def test_squat_fires_once_per_hold():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (45, dict(deltas=CROUCH_DOWN)),  # long hold: exactly one fire
        (12, seg_stand()),               # stand up -> re-arm
        (45, dict(deltas=CROUCH_DOWN)),
        (12, seg_stand()),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.CROUCH) == 2


def test_half_squat_fires_crouch():
    """A half squat (0.14 x torso sink, ~133 deg knees) is a real crouch."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    events = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (30, dict(deltas=CROUCH_LIGHT)),
        (12, seg_stand()),
    ]):
        events.extend(det.update(sample).events)
    assert count_actions(events, Action.CROUCH) == 1


# ---------------------------------------------------------------------------
# Running + state machine
# ---------------------------------------------------------------------------

def test_overlapping_jog_steps_still_start_running():
    """Pose smoothing may leave one leg high as the other crosses."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    both = {**RUN_LIFT_LEFT, **RUN_LIFT_RIGHT}
    snapshots = []
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (6, dict(deltas=RUN_LIFT_LEFT)),
        (6, dict(deltas=both)),             # R crosses while L still high
        (6, dict(deltas=RUN_LIFT_RIGHT)),   # L returns and re-arms
        (6, dict(deltas=both)),             # L crosses while R still high
    ]):
        snapshots.append(det.update(sample))
    assert snapshots[-1].steps >= 3
    assert snapshots[-1].running is True


def test_start_requires_three_alternating_steps():
    b = PoseBuilder(W_169, H_169)
    # Standing still never starts control (no steps registered).
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    run_motion(det, sm, b.stream([(40, seg_stand(0.85))]))
    assert sm.mode is ControllerMode.WAITING_FOR_RUN

    # Jogging in ANY lane starts control (CENTER no longer required).
    det2 = make_detector()
    sm2 = ControllerStateMachine()
    sm2.arm()
    transcript = run_motion(det2, sm2, b.stream(
        [(12, seg_stand(0.85)), *jog_segments(8, center_raw=0.85)],
    ))
    # Control started (stream may end in auto-pause once jogging stops,
    # which still proves ACTIVE was reached: the pause emits Esc).
    assert sm2.mode is not ControllerMode.WAITING_FOR_RUN
    assert sm2.mode is not ControllerMode.CALIBRATING
    # No key before ACTIVE, and ACTIVE only after the third accepted step.
    first_active = next(i for i, (_ts, s, _a) in enumerate(transcript)
                        if s.running)
    before = [a for _ts, _s, acts in transcript[:first_active] for a in acts]
    assert before == []


def test_waiting_for_run_suppresses_actions():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    transcript = run_motion(det, sm, b.stream([
        (12, seg_stand()),
        (18, dict(deltas=JUMP_UP)),  # jump while WAITING
        (14, seg_stand()),
        *jog_segments(6),
    ]))
    # ACTIVE was reached (a trailing auto-pause proves it happened).
    assert sm.mode is not ControllerMode.WAITING_FOR_RUN
    assert sm.mode is not ControllerMode.CALIBRATING
    acts = [a for _ts, _s, acts in transcript
            for a in acts if a is not Action.PAUSE_TOGGLE]
    assert acts == []


def test_leg_hover_between_thresholds_does_not_deadlock():
    """A leg resting between REARM and LIFT must not stay 'up' forever."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    # LIFT = 0.024 (0.12*0.20), REARM = 0.012 (0.06*0.20) with CAL.
    run_motion(det, sm, b.stream([
        (12, seg_stand()),
        (2, dict(deltas={25: (0.0, -0.03)})),    # left knee crosses the line
        (70, dict(deltas={25: (0.0, -0.018)})),  # hover between REARM and LIFT
        *jog_segments(8),
    ]))
    assert sm.mode is ControllerMode.ACTIVE


def test_hovering_leg_rearms_before_running_timeout():
    """A between-threshold landing cannot keep a leg latched for 2 seconds."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    for _ts, sample in b.stream([
        (12, seg_stand()),
        (2, dict(deltas={25: (0.0, -0.03)})),
        (24, dict(deltas={25: (0.0, -0.018)})),
    ]):
        det.update(sample)
    assert det._leg_up[25] is False


def test_rejected_lift_keeps_active_session_alive():
    """Visible jogging activity is liveness even when cadence rejects a step."""
    b = PoseBuilder(W_169, H_169)
    det, sm, active = run_active(b)
    before_steps = active[-1][1].steps
    same_lift = (RUN_LIFT_LEFT if det._last_step_side == 25
                 else RUN_LIFT_RIGHT)
    after = run_motion(det, sm, b.stream([
        (35, seg_stand()),
        (2, dict(deltas=same_lift)),  # too late/same-side: not a new step
        (25, seg_stand()),
    ], ts0=_after(active)))
    assert after[-1][1].steps == before_steps
    assert after[-1][1].running is True
    assert sm.mode is ControllerMode.ACTIVE


def test_three_sparse_steps_restart_prior_session():
    """After first activation, recovery counts fresh lifts without a 3 s gate."""
    b = PoseBuilder(W_169, H_169)
    det, sm, active = run_active(b)
    paused = run_motion(det, sm, b.stream([
        (90, seg_stand()),
    ], ts0=_after(active)))
    assert sm.mode is ControllerMode.AUTO_PAUSED
    paused_steps = paused[-1][1].steps

    recovered = run_motion(det, sm, b.stream([
        (2, dict(deltas=RUN_LIFT_RIGHT)),
        (160, seg_stand()),  # sequence expires; next lift starts a new chain
        (2, dict(deltas=RUN_LIFT_RIGHT)),
        (55, seg_stand()),
        (2, dict(deltas=RUN_LIFT_LEFT)),
        (12, seg_stand()),
    ], ts0=_after(paused)))
    assert recovered[-1][1].steps == paused_steps + 3
    assert any(snapshot.running for _ts, snapshot, _actions in recovered)
    resume_keys = [action for _ts, _snapshot, actions in recovered
                   for action in actions if action is Action.PAUSE_TOGGLE]
    assert resume_keys == [Action.PAUSE_TOGGLE]
    assert sm.mode is ControllerMode.ACTIVE


def test_stop_running_pauses_once_then_suppresses():
    b = PoseBuilder(W_169, H_169)
    det, sm, tr = run_active(b, jog_cycles=8)
    extra = run_motion(det, sm, b.stream([(90, seg_stand())], ts0=_after(tr)))
    pauses = [a for _ts, _s, acts in extra for a in acts
              if a is Action.PAUSE_TOGGLE]
    assert pauses == [Action.PAUSE_TOGGLE]
    assert sm.mode is ControllerMode.AUTO_PAUSED

    # Long stillness: exactly one pause total, no repeats.
    more = run_motion(det, sm, b.stream([(150, seg_stand())], ts0=_after(extra)))
    assert [a for _ts, _s, acts in more for a in acts] == []

    # Actions suppressed while paused, though the detector still fires.
    suppressed = run_motion(det, sm, b.stream([
        (18, dict(deltas=JUMP_UP)),
        (14, seg_stand()),
    ], ts0=_after(more)))
    assert any(Action.JUMP in s.events for _ts, s, _a in suppressed)
    assert [a for _ts, _s, acts in suppressed for a in acts] == []


def test_tick_without_new_frames_pauses_after_1_2s():
    b = PoseBuilder(W_169, H_169)
    det, sm, transcript = run_active(b, jog_cycles=8)
    last_ts = transcript[-1][0]
    actions = []
    now = last_ts
    for _ in range(120):  # camera silent; main thread ticks at 20 Hz
        now += 50
        actions.extend(sm.update(det.tick(now)))
    pauses = [a for a in actions if a is Action.PAUSE_TOGGLE]
    assert pauses == [Action.PAUSE_TOGGLE]
    assert sm.mode is ControllerMode.AUTO_PAUSED


def test_jump_grace_delays_pause_but_stays_bounded():
    b = PoseBuilder(W_169, H_169)
    det, sm, transcript = run_active(b, jog_cycles=8)
    last_step_ts = max(ts for ts, s, _ in transcript if s.running)
    # Jump once, land, then stand still forever (pose stays visible).
    after = run_motion(det, sm, b.stream([
        (18, dict(deltas=JUMP_UP)),
        (200, seg_stand()),
    ], ts0=_after(transcript)))
    pauses = [(ts, a) for ts, _s, acts in after for a in acts
              if a is Action.PAUSE_TOGGLE]
    assert len(pauses) == 1
    pause_ts = pauses[0][0]
    # Pause comes later than the plain 1.2 s rule (grace of <= 0.8 s)...
    assert pause_ts - last_step_ts >= 1_500
    # ...but stays bounded.
    assert pause_ts - last_step_ts <= 3_500
    assert sm.mode is ControllerMode.AUTO_PAUSED

def test_resume_after_three_new_steps_in_any_lane():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    t1 = run_motion(det, sm, b.stream([
        (12, seg_stand()),
        *jog_segments(8),           # ACTIVE
        (90, seg_stand()),          # stop -> AUTO_PAUSED
    ]))
    assert sm.mode is ControllerMode.AUTO_PAUSED
    assert sm.game_lane is Lane.CENTER

    # Jogging in a different (RIGHT) lane resumes control: the physical
    # pause lane is no longer required.
    transcript = run_motion(det, sm, b.stream([
        (12, seg_stand(0.85)),
        *jog_segments(12, center_raw=0.85),
    ], ts0=_after(t1)))
    pauses = [ts for ts, _s, acts in transcript for a in acts
              if a is Action.PAUSE_TOGGLE]
    assert len(pauses) == 1
    assert sm.mode is ControllerMode.ACTIVE
    # The resume key fires only after ~3 new steps (~0.7 s of jogging).
    assert pauses[0] >= 600
    assert sm.game_lane is Lane.CENTER  # no events forwarded while paused

def test_off_center_pause_still_resumes_in_place():
    """Pausing off-center changes nothing: jog where you stand."""
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    before = run_motion(det, sm, b.stream([
        (12, seg_stand(0.85)),
        *jog_segments(8, center_raw=0.85),
        (90, seg_stand(0.85)),
    ]))
    assert sm.mode is ControllerMode.AUTO_PAUSED
    assert sm.game_lane is Lane.CENTER

    resumed = run_motion(det, sm, b.stream([
        *jog_segments(12, center_raw=0.85),
    ], ts0=_after(before)))
    resume_keys = [a for _ts, _snap, actions in resumed for a in actions
                   if a is Action.PAUSE_TOGGLE]
    assert resume_keys == [Action.PAUSE_TOGGLE]
    assert sm.mode is ControllerMode.ACTIVE


def test_unknown_physical_lane_does_not_deadlock_resume():
    """Three steps resume even if lane and detector-running verdict lag."""
    sm = ControllerStateMachine()
    sm.arm()

    def snapshot(ts: int, running: bool, steps: int) -> MotionSnapshot:
        return MotionSnapshot(
            timestamp_ms=ts,
            pose_valid=True,
            stable_lane=None,
            running=running,
            steps=steps,
        )

    assert sm.update(snapshot(0, True, 3)) == ()
    assert sm.mode is ControllerMode.ACTIVE
    assert sm.update(snapshot(2_100, False, 3)) == (Action.PAUSE_TOGGLE,)
    assert sm.mode is ControllerMode.AUTO_PAUSED

    assert sm.update(snapshot(2_500, False, 5)) == ()
    assert sm.mode is ControllerMode.AUTO_PAUSED
    assert sm.update(snapshot(2_800, False, 6)) == (Action.PAUSE_TOGGLE,)
    assert sm.mode is ControllerMode.RESUMING

    # Resume grace is long enough for detector.running to catch up, but
    # bounded: a truly motionless user will auto-pause again.
    assert sm.update(snapshot(3_100, False, 6)) == ()
    assert sm.mode is ControllerMode.ACTIVE
    assert sm.update(snapshot(4_000, False, 6)) == ()
    assert sm.mode is ControllerMode.ACTIVE
    assert sm.update(snapshot(6_101, False, 6)) == (Action.PAUSE_TOGGLE,)
    assert sm.mode is ControllerMode.AUTO_PAUSED


def test_resuming_grace_suppresses_then_forwards():
    b = PoseBuilder(W_169, H_169)
    det = make_detector()
    sm = ControllerStateMachine()
    sm.arm()
    t0 = run_motion(det, sm, b.stream([
        (12, seg_stand()),
        *jog_segments(8),   # ACTIVE
        (90, seg_stand()),  # AUTO_PAUSED
    ]))
    assert sm.mode is ControllerMode.AUTO_PAUSED

    # Continuous jog until the resume key fires (mode -> RESUMING); a jump
    # injected immediately after must fire inside the 300 ms grace and be
    # suppressed, while a later jump is forwarded once ACTIVE.
    plan = [dict(deltas=RUN_LIFT_LEFT if i % 2 == 0 else RUN_LIFT_RIGHT)
            for i in range(200)]
    resume_ts = None
    all_actions = []
    det_jump_before = 0
    ts = _after(t0)
    i = 0
    while i < len(plan) or resume_ts is None:
        if i < len(plan):
            snap = det.update(b.sample(ts, **plan[i]))
            i += 1
        else:
            snap = det.tick(ts)
        acts = sm.update(snap)
        all_actions.extend((ts, a) for a in acts)
        if (resume_ts is None
                and any(a is Action.PAUSE_TOGGLE for a in acts)):
            resume_ts = ts
            inject = [dict(deltas=JUMP_UP)] * 18 + [seg_stand()] * 14
            for k in range(36):
                if k % 12 < 6:
                    lift = (RUN_LIFT_LEFT if (k // 12) % 2 == 0
                            else RUN_LIFT_RIGHT)
                    inject.append(dict(deltas=lift))
                else:
                    inject.append(seg_stand())
            inject += [dict(deltas=JUMP_UP)] * 18 + [seg_stand()] * 14
            plan = inject
            i = 0
        if resume_ts is not None and ts - resume_ts < 300:
            det_jump_before += sum(1 for e in snap.events if e is Action.JUMP)
        if resume_ts is not None and ts > resume_ts + 6_000:
            break
        ts += 33

    assert resume_ts is not None
    # The jump fired inside the grace window was suppressed...
    assert det_jump_before >= 1
    forwarded = [(t, a) for t, a in all_actions if a is Action.JUMP]
    # ...only the later jump made it through, after the 300 ms RESUMING grace.
    assert len(forwarded) == 1
    assert forwarded[0][0] - resume_ts >= 300
