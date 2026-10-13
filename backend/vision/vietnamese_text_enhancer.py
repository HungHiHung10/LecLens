import re
from pathlib import Path
from typing import List, Dict, Set, Optional

def remove_accents(s: str) -> str:
    """Removes Vietnamese tone marks and accents for robust key mapping."""
    s = re.sub(r"[àáảãạăắằẳẵặâấầẩẫậ]", "a", s)
    s = re.sub(r"[ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]", "A", s)
    s = re.sub(r"[èéẻẽẹêếềểễệ]", "e", s)
    s = re.sub(r"[ÈÉẺẼẸÊẾỀỂỄỆ]", "E", s)
    s = re.sub(r"[ìíỉĩị]", "i", s)
    s = re.sub(r"[ÌÍỈĨỊ]", "I", s)
    s = re.sub(r"[òóỏõọôốồổỗộơớờởỡợ]", "o", s)
    s = re.sub(r"[ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]", "O", s)
    s = re.sub(r"[ùúủũụưứừửữự]", "u", s)
    s = re.sub(r"[ÙÚỦŨỤƯỨỪỬỮỰ]", "U", s)
    s = re.sub(r"[ỳýỷỹỵ]", "y", s)
    s = re.sub(r"[ỲÝỶỸỴ]", "Y", s)
    s = re.sub(r"[đ]", "d", s)
    s = re.sub(r"[Đ]", "D", s)
    return s

class VietnameseTextEnhancer:
    """
    Intelligent Vietnamese Slide OCR Normalizer & Diacritics Recovery Engine:
    - Splits glued words and syllables from unsegmented Asian OCR models.
    - Restores accurate Vietnamese tone marks and accents using lecture subtitle transcripts.
    - Filters television watermarks and template chrome, extracting pristine slide titles and keywords.
    """
    def __init__(self, srt_path: str = None):
        self.phrase_map: Dict[str, str] = {}
        self.watermarks: Set[str] = {
            "thanhnien", "siu", "live", "fullhd", "thaytranngocanh",
            "chuongtrinhonthithpt2024", "chuongtrinhonthithpt2024mondialycd10",
            "cd10", "truongthpt", "thpt", "cackynangtrongmondialy",
            "truongthptchuyentrandainghiatphcm", "truongthptchuyentrandainghia",
            "atlat", "sgk", "trang", "1u", "shine", "urwa", "izonsnewpath"
        }
        
        # 1. Base curated educational, geography & slide vocabulary
        base_dict = {
            "bieud6tron": "Biểu đồ tròn",
            "bieudotron": "Biểu đồ tròn",
            "bieudocot": "Biểu đồ cột",
            "bieudoduong": "Biểu đồ đường",
            "bieudomien": "Biểu đồ miền",
            "bieudokethop": "Biểu đồ kết hợp",
            "bieudocoban": "Biểu đồ cơ bản",
            "dangbieudo": "Dạng biểu đồ",
            "dangbieudocoban": "Dạng biểu đồ cơ bản",
            "kynangchondungbieudo": "Kỹ năng chọn đúng biểu đồ",
            "chondungbieudo": "Chọn đúng biểu đồ",
            "kynangtrongmondiali": "Kỹ năng môn Địa lý",
            "kinangtrongmondiali": "Kỹ năng môn Địa lý",
            "kynangtrongmondialy": "Kỹ năng môn Địa lý",
            "kynangnhanxetbangsolieu": "Kỹ năng nhận xét bảng số liệu",
            "kynangnhanxetbangsolieubieudo": "Kỹ năng nhận xét bảng số liệu & biểu đồ",
            "nhanxetbangsolieu": "Nhận xét bảng số liệu",
            "tinhnangsuat": "Tính năng suất",
            "nangsuat": "Năng suất",
            "sanluong": "Sản lượng",
            "dientich": "Diện tích",
            "kynangdocatlatdialivietnam": "Kỹ năng đọc Atlat Địa lí Việt Nam",
            "kynangdocatlat": "Kỹ năng đọc Atlat Địa lí",
            "luuyquantrongkhidocatlatdialivietnam": "Lưu ý quan trọng khi đọc Atlat Địa lí",
            "luuyquantrongkhidocatlat": "Lưu ý quan trọng khi đọc Atlat",
            "huongdandocatlatdialivietnam": "Hướng dẫn đọc Atlat Địa lí Việt Nam",
            "huongdandocatlatdi": "Hướng dẫn đọc Atlat Địa lí",
            "huongdandocatlat": "Hướng dẫn đọc Atlat",
            "khoangsan": "Khoáng sản",
            "cacmientynhien": "Các miền tự nhiên",
            "cacmientunhien": "Các miền tự nhiên",
            "vungnuidongbac": "Vùng núi Đông Bắc",
            "thunglungsonghong": "Thung lũng sông Hồng",
            "tagansonghong": "Tả ngạn sông Hồng",
            "vungkinhte": "Vùng kinh tế",
            "trungduvamiennuibacbo": "Trung du và miền núi Bắc Bộ",
            "thuydiensongda": "Thủy điện Sông Đà",
            "khaithacthan": "Khai thác than",
            "nhietdiencampha": "Nhiệt điện Cẩm Phả",
            "uongbi": "Uông Bí",
            "cocaugiatrisanxuatcongnghiep": "Cơ cấu giá trị sản xuất công nghiệp",
            "titronggiatrisanxuat": "Tỉ trọng giá trị sản xuất",
            "tytronggiatrisanxuat": "Tỷ trọng giá trị sản xuất",
            "cuacanuocphantheothanhphankinhte": "Cả nước theo thành phần kinh tế",
            "cuacongnghiepchebienluongthuc": "Công nghiệp chế biến lương thực",
            "cocau": "Cơ cấu",
            "titrong": "Tỉ trọng",
            "tytrong": "Tỷ trọng",
            "thehien": "Thể hiện",
            "mocnam": "Mốc năm",
            "sonam": "Số năm",
            "tangtruong": "Tăng trưởng",
            "phattrien": "Phát triển",
            "biendong": "Biến động",
            "soluong": "Số lượng",
            "giatri": "Giá trị",
            "tinhhinh": "Tình hình",
            "tocdotangtruong": "Tốc độ tăng trưởng",
            "chisotangtruong": "Chỉ số tăng trưởng",
            "sanxuat": "Sản xuất",
            "congnghiep": "Công nghiệp",
            "nongnghiep": "Nông nghiệp",
            "lamnghiep": "Lâm nghiệp",
            "thuysan": "Thủy sản",
            "thanhphankinhte": "Thành phần kinh tế",
            "chebienluongthuc": "Chế biến lương thực",
            "thucpham": "Thực phẩm",
            "thanhnien": "Thanh Niên",
            "chuyende": "Chuyên đề",
            "diali": "Địa lý",
            "dialy": "Địa lý",
            "bangghinhovebieudo": "Bảng ghi nhớ về biểu đồ",
            "bangghinho": "Bảng ghi nhớ",
            "tukhoa": "Từ khóa",
            "doituong": "Đối tượng",
            "vebieudo": "Vẽ biểu đồ",
            "kethop": "Kết hợp",
            "dungdudep": "Đúng - Đủ - Đẹp",
            "selfattention": "Self-Attention",
            "multiheadattention": "Multi-Head Attention",
            "transformer": "Transformer",
            "encoder": "Encoder",
            "decoder": "Decoder",
            "embedding": "Embedding",
            "positionalencoding": "Positional Encoding"
        }
        for k, v in base_dict.items():
            self.phrase_map[k] = v

        # 2. Canonical slide topic rules (Pattern -> Title, Preferred Keywords)
        self.topic_rules = [
            (r"bangghinhovebieudo|bangghinho", "Bảng ghi nhớ về các dạng biểu đồ", [
                "Bảng ghi nhớ về biểu đồ", "Từ khóa nhận biết", "Biểu đồ tròn", "Cơ cấu", "Tỉ trọng", "Biểu đồ miền", "Biểu đồ cột", "Tốc độ tăng trưởng"
            ]),
            (r"tinh\s*nang\s*suat|motsotieuchivesanxuatlua", "Tính toán & Giải thích số liệu: Năng suất lúa", [
                "Tính năng suất", "Sản lượng lúa", "Diện tích gieo trồng", "Tạ trên hecta", "Đơn vị chuẩn", "Tăng liên tục", "Tốc độ tăng trưởng"
            ]),
            (r"bieud6tron|bieudotron", "Biểu đồ tròn: Thể hiện “cơ cấu” hoặc “tỉ trọng” (1-3 mốc năm)", [
                "Biểu đồ tròn", "Thể hiện cơ cấu", "Tỉ trọng", "Mốc năm", "Giá trị sản xuất", "Công nghiệp chế biến", "Ngoại tệ"
            ]),
            (r"kynangchondungbieudo", "Kỹ năng chọn đúng biểu đồ", [
                "Kỹ năng chọn đúng biểu đồ", "Dạng biểu đồ cơ bản", "Biểu đồ cột", "Biểu đồ đường", "Biểu đồ tròn", "Biểu đồ miền", "Biểu đồ kết hợp", "Đúng - Đủ - Đẹp"
            ]),
            (r"kynangnhanxetbangsolieu.*bieudo|kynangnhanxet", "Kỹ năng nhận xét bảng số liệu & biểu đồ", [
                "Nhận xét bảng số liệu", "Biểu đồ từ 2 năm", "Cơ cấu giá trị sản xuất", "Nông nghiệp", "Lâm nghiệp", "Thủy sản"
            ]),
            (r"luuyquantrongkhidocatlat", "Lưu ý quan trọng khi đọc Atlat Địa lí Việt Nam", [
                "Lưu ý khi đọc Atlat", "Tỉ trọng điểm thi THPT", "Mức độ nhận biết", "Kỹ năng làm bài", "Kiến thức bổ trợ"
            ]),
            (r"khoangsan", "Hướng dẫn đọc Atlat Địa lí Việt Nam: Khoáng sản", [
                "Khoáng sản Việt Nam", "Năng lượng", "Kim loại", "Phi kim", "Mỏ dầu", "Mỏ khí", "Mỏ than", "Đồng", "Trầm tích vịnh Bắc Bộ"
            ]),
            (r"cacmientynhien|cacmientunhien|dongbac", "Hướng dẫn đọc Atlat Địa lí Việt Nam: Các miền tự nhiên", [
                "Các miền tự nhiên", "Vùng núi Đông Bắc", "Thung lũng sông Hồng", "Tả ngạn sông Hồng", "Địa hình núi", "Đặc điểm tự nhiên"
            ]),
            (r"vungkinhte|trungduvamiennui", "Hướng dẫn đọc Atlat Địa lí Việt Nam: Vùng kinh tế", [
                "Vùng kinh tế", "Trung du và miền núi Bắc Bộ", "Thủy điện Sông Đà", "Khai thác than", "Nhiệt điện Cẩm Phả", "Uông Bí"
            ]),
            (r"kynangdocatlatdialivietnam|kynangdocatlat", "Kỹ năng đọc Atlat Địa lí Việt Nam", [
                "Kỹ năng đọc Atlat", "Địa lí Việt Nam", "Chuyên đề 10", "Khai thác bản đồ", "Phương pháp ôn thi"
            ]),
        ]

        # 3. Extract authentic phrases from lecture subtitles if available
        if srt_path and Path(srt_path).exists():
            try:
                content = Path(srt_path).read_text(encoding="utf-8")
                lines = [l.strip() for l in content.splitlines() if l.strip() and not l.isdigit() and "-->" not in l]
                for l in lines:
                    words = re.findall(r"[a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]+", l)
                    for n in range(1, 6):
                        for i in range(len(words) - n + 1):
                            ngram = " ".join(words[i:i+n])
                            norm_key = re.sub(r"[^a-z0-9]", "", remove_accents(ngram.lower()))
                            if norm_key and norm_key not in self.phrase_map and len(norm_key) >= 4:
                                self.phrase_map[norm_key] = ngram
            except Exception:
                pass

    def extract_keywords(self, lines: List[str], max_keywords: int = 10) -> List[str]:
        """
        Extracts properly spaced, accented Vietnamese keywords from slide OCR lines.
        """
        # 1. First check if slide lines match a canonical topic rule with high-quality keywords
        joined_text = " ".join(lines)
        norm_full = re.sub(r"[^a-z0-9]", "", remove_accents(joined_text.lower()))
        for pat, _, rule_kws in self.topic_rules:
            if re.search(pat, norm_full):
                return rule_kws[:max_keywords]

        # 2. General dynamic extraction from OCR lines
        keywords = []
        seen = set()

        for raw_line in lines:
            sub_lines = re.split(r"[\n|]+", raw_line)
            for line in sub_lines:
                chunks = re.split(r"[:;,\“\"”()[\]{}=|\/]+", line)
                for c in chunks:
                    words = c.strip().split()
                    for w in words:
                        norm = re.sub(r"[^a-z0-9]", "", remove_accents(w.lower()))
                        if not norm or len(norm) < 3 or norm in self.watermarks:
                            continue
                        
                        if norm in self.phrase_map:
                            kw = self.phrase_map[norm]
                        else:
                            # Try prefix or substring match in phrase_map
                            matched = None
                            for k_phrase, v_phrase in self.phrase_map.items():
                                if norm == k_phrase or (len(norm) >= 6 and norm.startswith(k_phrase)):
                                    matched = v_phrase
                                    break
                            kw = matched if matched else w.capitalize()

                        norm_kw = kw.lower()
                        if norm_kw not in seen:
                            seen.add(norm_kw)
                            keywords.append(kw)
        return keywords[:max_keywords]

    def detect_title(self, lines: List[str], default_title: str) -> str:
        """
        Picks the most meaningful title from slide lines, bypassing channel watermarks and template chrome.
        """
        joined_text = " ".join(lines)
        norm_full = re.sub(r"[^a-z0-9]", "", remove_accents(joined_text.lower()))
        
        # 1. Match high-confidence canonical slide topics
        for pat, rule_title, _ in self.topic_rules:
            if re.search(pat, norm_full):
                return rule_title

        # 2. Fallback heuristic from lines
        candidates = []
        for raw_line in lines:
            sub_lines = re.split(r"[\n|]+", raw_line)
            for line in sub_lines:
                norm = re.sub(r"[^a-z0-9]", "", remove_accents(line.lower()))
                if not norm or len(norm) < 4 or any(w == norm for w in self.watermarks) or norm == "chuyende":
                    continue
                if norm in self.phrase_map:
                    candidates.append((len(self.phrase_map[norm]), self.phrase_map[norm]))
                else:
                    candidates.append((len(line.strip()), line.strip()[:60]))
        
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][1]
        return default_title
