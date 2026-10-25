"""Decide when a spoken answer starts and ends, from microphone blocks.

Energy based: the first blocks measure the room's noise floor, speech is anything well above it,
and an answer ends after a stretch of silence. Pure numpy, so it is tested without a microphone;
Whisper's own VAD filter cleans up whatever noise gets through.
"""

import enum
from dataclasses import dataclass, field

import numpy as np

# Speech must be this many times louder (RMS) than the noise floor: about +10 dB.
SPEECH_RATIO = 3.0
# Never treat anything quieter than this as speech, however quiet the room (~ -46 dBFS).
MIN_SPEECH_RMS = 0.005


class State(enum.Enum):
    WAITING = "waiting"  # no speech yet
    SPEAKING = "speaking"
    DONE = "done"  # speech, then enough silence (or the length cap)
    NO_SPEECH = "no_speech"  # nothing said before the listen timeout


@dataclass
class Endpointer:
    sample_rate: int
    listen_timeout: float  # seconds to wait for speech to start
    end_silence: float  # silence that ends the answer, seconds
    max_answer: float  # longest answer, seconds
    calibration: float = 0.3  # seconds of audio used to measure the noise floor
    min_speech: float = 0.2  # loud stretch that counts as speech starting, seconds

    state: State = field(default=State.WAITING, init=False)
    _elapsed: float = field(default=0.0, init=False)
    _noise: list[float] = field(default_factory=list, init=False)
    _threshold: float = field(default=MIN_SPEECH_RMS, init=False)
    _loud_run: float = field(default=0.0, init=False)
    _silent_run: float = field(default=0.0, init=False)
    _speech_started_at: float = field(default=0.0, init=False)

    def feed(self, block: np.ndarray) -> State:
        """Consume one block of mono float samples and return the new state."""
        if self.state in (State.DONE, State.NO_SPEECH) or block.size == 0:
            return self.state
        duration = block.size / self.sample_rate
        self._elapsed += duration
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))

        if self._elapsed <= self.calibration:
            self._noise.append(rms)
            floor = float(np.median(self._noise))
            self._threshold = max(MIN_SPEECH_RMS, floor * SPEECH_RATIO)
            return self.state

        loud = rms >= self._threshold
        if self.state is State.WAITING:
            self._loud_run = self._loud_run + duration if loud else 0.0
            if self._loud_run >= self.min_speech:
                self.state = State.SPEAKING
                self._speech_started_at = self._elapsed - self._loud_run
                self._silent_run = 0.0
            elif self._elapsed >= self.listen_timeout:
                self.state = State.NO_SPEECH
            return self.state

        self._silent_run = 0.0 if loud else self._silent_run + duration
        too_long = self._elapsed - self._speech_started_at >= self.max_answer
        if self._silent_run >= self.end_silence or too_long:
            self.state = State.DONE
        return self.state
