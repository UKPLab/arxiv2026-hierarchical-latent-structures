from typing import Union
import pendulum


def format_duration(seconds: Union[int, float]) -> str:
    duration = pendulum.duration(seconds=int(seconds))
    return duration.in_words(locale="en")


def format_duration_from_timestamps(start: str, end: str) -> str:
    start_time = pendulum.parse(start)
    end_time = pendulum.parse(end)
    duration = end_time - start_time
    return duration.in_words(locale="en")