"""
Seed script to pre-populate VocabMaster with initial rich vocabulary.
"""
from app.database import init_db, save_words, get_stats

SAMPLE_WORDS = [
    {
        "word": "accommodate",
        "ipa": "/əˈkɒm.ə.deɪt/",
        "pos": "verb",
        "meaning_vi": "đáp ứng, điều chỉnh để phù hợp với nhu cầu",
        "example_sentence": "The hotel was able to accommodate all the international delegates during the summit.",
        "distractors": ["phân chia tài sản", "từ chối hợp tác", "chậm trễ tiến độ"],
        "cefr_level": "B2"
    },
    {
        "word": "anticipate",
        "ipa": "/ænˈtɪs.ɪ.peɪt/",
        "pos": "verb",
        "meaning_vi": "dự đoán, lường trước tình huống",
        "example_sentence": "We did not anticipate having such a large turnout for the workshop.",
        "distractors": ["lặp lại nhiều lần", "bỏ qua chi tiết", "kết thúc sớm"],
        "cefr_level": "B2"
    },
    {
        "word": "collaborate",
        "ipa": "/kəˈlæb.ə.reɪt/",
        "pos": "verb",
        "meaning_vi": "hợp tác, phối hợp chặt chẽ",
        "example_sentence": "Researchers from several universities will collaborate on the climate change project.",
        "distractors": ["cạnh tranh gay gắt", "đơn phương thực hiện", "giấu thông tin"],
        "cefr_level": "B2"
    },
    {
        "word": "comprehensive",
        "ipa": "/ˌkɒm.prɪˈhen.sɪv/",
        "pos": "adjective",
        "meaning_vi": "toàn diện, bao quát mọi khía cạnh",
        "example_sentence": "The report provides a comprehensive overview of global economic trends.",
        "distractors": ["nông cạn, sơ sài", "rời rạc từng phần", "tạm thời, ngắn hạn"],
        "cefr_level": "B2"
    },
    {
        "word": "crucial",
        "ipa": "/ˈkruː.ʃəl/",
        "pos": "adjective",
        "meaning_vi": "cốt yếu, mang tính quyết định",
        "example_sentence": "Early detection of the disease is crucial for effective medical treatment.",
        "distractors": ["không đáng kể", "thứ yếu, phụ trợ", "tùy ý lựa chọn"],
        "cefr_level": "B2"
    },
    {
        "word": "demonstrate",
        "ipa": "/ˈdem.ən.streɪt/",
        "pos": "verb",
        "meaning_vi": "chứng minh, làm sáng tỏ qua ví dụ thực tế",
        "example_sentence": "The experiment results clearly demonstrate the validity of our initial theory.",
        "distractors": ["phủ định sạch trơn", "nghi ngờ kết quả", "che giấu bằng chứng"],
        "cefr_level": "B2"
    },
    {
        "word": "enhance",
        "ipa": "/ɪnˈhɑːns/",
        "pos": "verb",
        "meaning_vi": "nâng cao, tăng cường chất lượng hoặc giá trị",
        "example_sentence": "Listening to English podcasts regularly will significantly enhance your listening comprehension.",
        "distractors": ["làm suy thoái", "cản trở tiến bộ", "giữ nguyên hiện trạng"],
        "cefr_level": "B2"
    },
    {
        "word": "fundamental",
        "ipa": "/ˌfʌn.dəˈmen.təl/",
        "pos": "adjective",
        "meaning_vi": "cơ bản, nền tảng cốt lõi",
        "example_sentence": "Understanding grammar rules is a fundamental part of mastering a new language.",
        "distractors": ["bên ngoài, bề mặt", "vụn vặt không quan trọng", "phức tạp quá mức"],
        "cefr_level": "B2"
    },
    {
        "word": "implement",
        "ipa": "/ˈɪm.plɪ.ment/",
        "pos": "verb",
        "meaning_vi": "triển khai thực hiện, áp dụng thực tế",
        "example_sentence": "The government decided to implement strict new regulations on data privacy.",
        "distractors": ["hủy bỏ kế hoạch", "trì hoãn quyết định", "thảo luận lý thuyết"],
        "cefr_level": "B2"
    },
    {
        "word": "innovative",
        "ipa": "/ˈɪn.ə.və.tɪv/",
        "pos": "adjective",
        "meaning_vi": "sáng tạo, mang tính đổi mới đột phá",
        "example_sentence": "The startup gained rapid market traction thanks to its innovative product design.",
        "distractors": ["lạc hậu, cũ kỹ", "rập khuôn máy móc", "bắt chước nguyên xi"],
        "cefr_level": "B2"
    },
    {
        "word": "perspective",
        "ipa": "/pəˈspek.tɪv/",
        "pos": "noun",
        "meaning_vi": "góc nhìn, quan điểm đánh giá",
        "example_sentence": "Traveling abroad gives students a broader perspective on cultural diversity.",
        "distractors": ["sự thiên vị cá nhân", "suy nghĩ hẹp hòi", "định kiến xã hội"],
        "cefr_level": "B2"
    },
    {
        "word": "resilient",
        "ipa": "/rɪˈzɪl.jənt/",
        "pos": "adjective",
        "meaning_vi": "kiên cường, nhanh chóng phục hồi sau khó khăn",
        "example_sentence": "Local businesses proved remarkably resilient throughout the economic downturn.",
        "distractors": ["mong manh, dễ vỡ", "nhanh chóng bỏ cuộc", "bị động hoàn toàn"],
        "cefr_level": "C1"
    },
    {
        "word": "sustain",
        "ipa": "/səˈsteɪn/",
        "pos": "verb",
        "meaning_vi": "duy trì lâu dài, chống đỡ vững vàng",
        "example_sentence": "The company struggled to sustain its high growth rate over several quarters.",
        "distractors": ["bỏ cuộc giữa chừng", "làm gián đoạn", "cắt giảm ngân sách"],
        "cefr_level": "C1"
    },
    {
        "word": "utilize",
        "ipa": "/ˈjuː.təl.aɪz/",
        "pos": "verb",
        "meaning_vi": "tận dụng, sử dụng một cách hiệu quả",
        "example_sentence": "You should utilize modern software tools to automate repetitive tasks.",
        "distractors": ["lãng phí tài nguyên", "bỏ quên không dùng", "phá hoại công cụ"],
        "cefr_level": "B2"
    }
]

if __name__ == "__main__":
    init_db()
    saved = save_words(None, SAMPLE_WORDS)
    print(f"Successfully seeded {len(saved)} words!")
    print(get_stats())
