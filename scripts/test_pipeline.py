import re
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

def remove_accents(s: str) -> str:
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
    def __init__(self, srt_path: str = None):
        self.phrase_map = {}
        
        # 1. Base dictionary for common slide words
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
            "cocaugiatrisanxuatcongnghiep": "Cơ cấu giá trị sản xuất công nghiệp",
            "titronggiatrisanxuat": "Tỉ trọng giá trị sản xuất",
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
            "tocdotangtruong": "Tốc độ tăng trưởng",
            "chisotangtruong": "Chỉ số tăng trưởng",
            "sanxuat": "Sản xuất",
            "congnghiep": "Công nghiệp",
            "thanhphankinhte": "Thành phần kinh tế",
            "chebienluongthuc": "Chế biến lương thực",
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
            "dungdudep": "Đúng - Đủ - Đẹp"
        }
        for k, v in base_dict.items():
            self.phrase_map[k] = v

        # 2. Extract from SRT subtitles if present
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
            except Exception as e:
                pass

    def enhance_token(self, tok: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]", "", tok).lower()
        if not clean:
            return tok
        if clean in self.phrase_map:
            return self.phrase_map[clean]
        return tok

    def extract_keywords(self, lines: list) -> list:
        keywords = []
        seen = set()
        watermarks = {"thanhnien", "siu", "live", "fullhd", "thaytranngocanh", "chuongtrinhonthithpt2024", "cd10", "truongthpt"}

        for line in lines:
            # Split by punctuation
            chunks = re.split(r"[:;,\“\"”()[\]{}=|\/]+", line)
            for c in chunks:
                words = c.strip().split()
                for w in words:
                    norm = re.sub(r"[^a-z0-9]", "", remove_accents(w.lower()))
                    if not norm or len(norm) < 3 or norm in watermarks:
                        continue
                    
                    if norm in self.phrase_map:
                        kw = self.phrase_map[norm]
                    else:
                        kw = w.capitalize()

                    norm_kw = kw.lower()
                    if norm_kw not in seen:
                        seen.add(norm_kw)
                        keywords.append(kw)
        return keywords[:8]

    def detect_title(self, lines: list, default_title: str) -> str:
        watermarks = {"thanhnien", "siu", "live", "fullhd", "thpt", "chuyende"}
        candidates = []
        for line in lines:
            norm = re.sub(r"[^a-z0-9]", "", remove_accents(line.lower()))
            if not norm or len(norm) < 4 or any(w == norm for w in watermarks):
                continue
            if norm in self.phrase_map:
                candidates.append((len(self.phrase_map[norm]), self.phrase_map[norm]))
            else:
                candidates.append((len(line.strip()), line.strip()[:50]))
        
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][1]
        return default_title

enhancer = VietnameseTextEnhancer("data/uploads/custom_1791461236.mp4.srt")
meta = json.load(open("data/keyframes/custom_1791461236/keyframes_metadata.json", "r", encoding="utf-8"))

print("=== FINAL REFINED COMPARISON ===")
for idx in [4, 5, 6]:
    kf = meta[idx]
    lines = [l.strip() for l in kf["ocr_text"].split("|") if l.strip()]
    title = enhancer.detect_title(lines, kf["slide_title"])
    kw = enhancer.extract_keywords(lines)
    print(f"\n[Slide #{idx} @ {kf['timestamp_str']}]")
    print(f"  Title   : {title}")
    print(f"  Keywords: {', '.join(kw)}")
