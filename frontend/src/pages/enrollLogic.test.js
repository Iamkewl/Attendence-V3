import { describe, expect, it } from 'vitest'
import {
  POSE_SEQUENCE,
  advancePose,
  decideOk,
  initialPoseMachine,
  isHardError,
  poseDone,
} from './enrollLogic'

describe('decideOk', () => {
  it('is true only for a detected face with an empty reasons array', () => {
    expect(decideOk({ detected: true, num_faces: 1, bbox: [0.3, 0.2, 0.4, 0.5], reasons: [] })).toBe(true)
  })

  it('is false when the server reports any diagnostic reason', () => {
    expect(decideOk({ detected: true, num_faces: 1, reasons: ['MOTION_BLUR'] })).toBe(false)
  })

  it('is false when nothing was detected even with empty reasons', () => {
    expect(decideOk({ detected: false, num_faces: 0, reasons: [] })).toBe(false)
    expect(decideOk(null)).toBe(false)
  })
})

describe('pose state machine', () => {
  it('requires two consecutive ok previews to complete a pose', () => {
    const once = advancePose(initialPoseMachine(), true)
    expect(once.consecutiveOk).toBe(1)
    expect(once.completedCount).toBe(0)

    const twice = advancePose(once, true)
    expect(twice.consecutiveOk).toBe(0)
    expect(twice.completedCount).toBe(1)
    expect(twice.poseIndex).toBe(1)
  })

  it('resets the streak on a non-ok preview', () => {
    let machine = advancePose(initialPoseMachine(), true)
    machine = advancePose(machine, false)
    expect(machine.consecutiveOk).toBe(0)
    expect(machine.completedCount).toBe(0)
  })

  it('walks the full three-pose sequence after six consecutive oks', () => {
    let machine = initialPoseMachine()
    for (let i = 0; i < 6; i += 1) machine = advancePose(machine, true)
    expect(poseDone(machine)).toBe(true)

    // The completed sequence maps onto the full pose list.
    expect(POSE_SEQUENCE).toHaveLength(3)
  })

  it('ignores previews once the sequence is done', () => {
    let machine = initialPoseMachine()
    for (let i = 0; i < 6; i += 1) machine = advancePose(machine, true)

    const afterBad = advancePose(machine, false)
    expect(afterBad).toBe(machine)
  })
})

describe('isHardError', () => {
  it('flags auth, role, size and availability failures as hard errors', () => {
    for (const status of [401, 403, 413, 503]) {
      expect(isHardError({ response: { status } })).toBe(true)
    }
  })

  it('treats quality refusals and network errors as non-hard', () => {
    expect(isHardError({ response: { status: 422 } })).toBe(false)
    expect(isHardError(new Error('network down'))).toBe(false)
  })
})
