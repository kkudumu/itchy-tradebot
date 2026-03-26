"""
Trading session identification for XAU/USD.

All session boundaries are in UTC. Gold trading volume and volatility
peaks during the London/NY overlap (13:00–16:00 UTC).
"""

from __future__ import annotations

import numpy as np


# Session boundary tuples: (start_hour, start_minute, end_hour, end_minute)
_SESSION_BOUNDS: dict[str, tuple[int, int, int, int]] = {
    "asian":    (0,  0,  8,  0),   # 00:00–08:00 UTC
    "london":   (8,  0, 16,  0),   # 08:00–16:00 UTC
    "new_york": (13, 0, 21,  0),   # 13:00–21:00 UTC
    "overlap":  (13, 0, 16,  0),   # 13:00–16:00 UTC (London/NY overlap)
}


def _minutes_of_day(timestamps: np.ndarray) -> np.ndarray:
    """Convert an array of UTC timestamps to minutes since midnight (0–1439).

    Parameters
    ----------
    timestamps:
        numpy array of ``numpy.datetime64`` values (any resolution).

    Returns
    -------
    np.ndarray of int32 minutes since midnight.
    """
    # Convert to seconds since epoch, then extract time-of-day component
    ts_sec = timestamps.astype("datetime64[s]").astype(np.int64)
    seconds_in_day = 86_400
    seconds_today = ts_sec % seconds_in_day
    # Guard against negative modulo on pre-epoch timestamps
    seconds_today = np.where(seconds_today < 0, seconds_today + seconds_in_day, seconds_today)
    return (seconds_today // 60).astype(np.int32)


def _session_start_end_minutes(session: str) -> tuple[int, int]:
    """Return (start_minutes, end_minutes) for the given session key."""
    sh, sm, eh, em = _SESSION_BOUNDS[session]
    return sh * 60 + sm, eh * 60 + em


class SessionIdentifier:
    """Identify forex/gold trading sessions from UTC timestamp arrays.

    Session priority when overlapping:
        overlap > london / new_york > asian > "off_hours"

    The overlap session (13:00–16:00 UTC) is the highest-priority label
    because it represents peak gold liquidity.
    """

    # Expose bounds for external inspection / testing
    SESSIONS = _SESSION_BOUNDS

    # Priority order (highest first) for label assignment
    _PRIORITY: list[str] = ["overlap", "london", "new_york", "asian"]

    def identify(self, timestamps: np.ndarray) -> np.ndarray:
        """Return a string session label for each timestamp.

        Parameters
        ----------
        timestamps:
            1-D array of ``numpy.datetime64`` UTC timestamps.

        Returns
        -------
        np.ndarray of dtype ``object`` (strings):
            One of ``"overlap"``, ``"london"``, ``"new_york"``, ``"asian"``,
            or ``"off_hours"`` for each element.
        """
        timestamps = np.asarray(timestamps, dtype="datetime64[s]")
        minutes = _minutes_of_day(timestamps)

        labels = np.full(len(timestamps), "off_hours", dtype=object)

        # Assign in reverse-priority order so higher-priority sessions overwrite
        for session in reversed(self._PRIORITY):
            mask = self.session_mask(timestamps, session)
            labels[mask] = session

        return labels

    def is_active_session(self, timestamps: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating prime gold trading hours.

        Active is defined as London or NY session (08:00–21:00 UTC),
        which covers the 08:00–17:00 GMT window referenced in gold trading
        literature, extended to capture the full NY afternoon session.

        Parameters
        ----------
        timestamps:
            1-D array of ``numpy.datetime64`` UTC timestamps.

        Returns
        -------
        np.ndarray of bool.
        """
        timestamps = np.asarray(timestamps, dtype="datetime64[s]")
        return self.session_mask(timestamps, "london") | self.session_mask(timestamps, "new_york")

    def session_mask(self, timestamps: np.ndarray, session: str) -> np.ndarray:
        """Return a boolean mask for a specific named session.

        Parameters
        ----------
        timestamps:
            1-D array of ``numpy.datetime64`` UTC timestamps.
        session:
            One of ``"asian"``, ``"london"``, ``"new_york"``, ``"overlap"``.

        Returns
        -------
        np.ndarray of bool.

        Raises
        ------
        ValueError
            If ``session`` is not a recognised session name.
        """
        if session not in _SESSION_BOUNDS:
            raise ValueError(
                f"Unknown session '{session}'. "
                f"Valid options: {list(_SESSION_BOUNDS.keys())}"
            )

        timestamps = np.asarray(timestamps, dtype="datetime64[s]")
        minutes = _minutes_of_day(timestamps)
        start, end = _session_start_end_minutes(session)
        return (minutes >= start) & (minutes < end)
