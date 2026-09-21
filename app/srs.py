"""
SuperMemo-2 (SM-2) Spaced Repetition Algorithm Implementation.
"""
from datetime import datetime, timedelta, timezone

MAX_INTERVAL_DAYS = 365
MIN_EASE_FACTOR = 1.3
MAX_EASE_FACTOR = 3.0

def calculate_sm2(quality: int, repetitions: int, ease_factor: float, interval: int):
    """
    quality: Đánh giá nhớ từ (0 - 5)
      0: Hoàn toàn không nhớ (Blackout)
      1: Sai, nhưng nhìn đáp án thấy quen
      2: Sai, nhưng nhớ mang máng
      3: Đúng nhưng rất khó khăn (Hard)
      4: Đúng sau chút do dự (Good)
      5: Nhớ hoàn hảo, phản xạ ngay (Easy)
    repetitions: Số lần liên tiếp trả lời đúng
    ease_factor: Hệ số dễ nhớ (mặc định 2.5, min 1.3, max 3.0)
    interval: Khoảng cách ngày của lần ôn trước
    
    Trả về: dict chứa (interval, repetitions, ease_factor, due_date)
    """
    try:
        quality = max(0, min(5, int(quality)))
    except (ValueError, TypeError):
        quality = 0

    try:
        repetitions = max(0, int(repetitions or 0))
    except (ValueError, TypeError):
        repetitions = 0

    try:
        ease_factor = float(ease_factor or 2.5)
    except (ValueError, TypeError):
        ease_factor = 2.5

    try:
        interval = max(1, int(interval or 1))
    except (ValueError, TypeError):
        interval = 1
    
    if quality >= 3:
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = max(1, int(round(interval * ease_factor)))
        new_repetitions = repetitions + 1
    else:
        new_repetitions = 0
        new_interval = 1

    # Chặn trần tối đa cho khoảng cách ngày ôn tập
    new_interval = min(MAX_INTERVAL_DAYS, new_interval)

    # Công thức điều chỉnh hệ số dễ nhớ EF
    new_ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if new_ease_factor < MIN_EASE_FACTOR:
        new_ease_factor = MIN_EASE_FACTOR
    elif new_ease_factor > MAX_EASE_FACTOR:
        new_ease_factor = MAX_EASE_FACTOR

    # Làm tròn về 00:00:00 UTC của ngày đến hạn để tránh hiện tượng Interval Drift
    now_utc = datetime.now(timezone.utc)
    target_date = now_utc.date() + timedelta(days=new_interval)
    due_date = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=timezone.utc)
    
    return {
        "interval": new_interval,
        "repetitions": new_repetitions,
        "ease_factor": round(new_ease_factor, 2),
        "due_date": due_date.isoformat()
    }
