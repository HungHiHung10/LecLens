import re
import json
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Common Vietnamese unaccented syllables
VN_SYLLABLES = {
    "an", "anh", "ao", "ap", "at", "au", "ay",
    "ba", "bac", "ban", "bang", "bao", "bap", "bat", "bay", "be", "ben", "beng", "beo", "bi", "biec", "bien", "biet", "bieu", "binh", "bo", "boc", "boi", "bom", "bon", "bong", "bot", "bu", "bua", "bui", "bun", "buoc", "buon", "buot",
    "ca", "cac", "cai", "cam", "can", "cang", "canh", "cao", "cap", "cat", "cau", "cay", "ce", "cha", "chac", "chai", "cham", "chan", "chang", "chanh", "chao", "chap", "chat", "chau", "chay", "che", "chen", "cheo", "chi", "chia", "chiec", "chiem", "chien", "chieu", "chin", "chinh", "cho", "choc", "choi", "chon", "chong", "chot", "chu", "chua", "chuc", "chui", "chum", "chun", "chung", "chuoc", "chuoi", "chuon", "chuot", "chup", "chuyen", "chuyn", "co", "coc", "coi", "com", "con", "cong", "cot", "cu", "cua", "cuc", "cui", "cum", "cung", "cuoc", "cuoi", "cuon", "cuop",
    "da", "dac", "dai", "dam", "dan", "dang", "danh", "dao", "dap", "dat", "dau", "day", "de", "dem", "den", "deoc", "deu", "di", "dia", "diec", "diem", "dien", "diep", "diet", "dieu", "dinh", "do", "doan", "doc", "doi", "don", "dong", "dot", "du", "dua", "duc", "due", "dui", "dum", "dung", "duoc", "duoi", "duon", "duot", "duyen",
    "em", "en", "eo",
    "ga", "gac", "gai", "gam", "gan", "gang", "ganh", "gao", "gap", "gat", "gay", "ge", "gen", "ghi", "gia", "giac", "giai", "giam", "gian", "giang", "gianh", "giao", "giap", "giat", "giau", "giay", "gie", "gieng", "gio", "gioi", "giom", "gion", "giong", "giot", "giu", "giua", "giuc", "giup", "go", "goc", "goi", "gom", "gon", "gong", "got", "gu", "gua", "guc", "gui", "gum", "gung",
    "ha", "hac", "hai", "ham", "han", "hang", "hanh", "hao", "hap", "hat", "hau", "hay", "he", "hem", "hen", "heo", "hi", "hiep", "hieu", "hin", "hinh", "ho", "hoa", "hoac", "hoai", "hoan", "hoang", "hoap", "hoat", "hoc", "hoi", "hom", "hon", "hong", "hop", "hot", "hu", "hua", "huc", "hue", "hui", "hum", "hung", "huoc", "huong", "huot", "hup", "huy", "huyen", "huynh",
    "i", "ich", "iem", "ien", "iep", "iet", "ieu", "in", "inh", "it",
    "ke", "kem", "ken", "keo", "ket", "kha", "khac", "khai", "kham", "khan", "khang", "khanh", "khao", "khat", "khau", "khay", "khe", "khen", "kheo", "khi", "khia", "khiec", "khien", "khiep", "khiet", "khieu", "khin", "khinh", "kho", "khoa", "khoac", "khoai", "khoan", "khoang", "khoat", "khoc", "khoi", "khom", "khon", "khong", "khop", "khu", "khua", "khuc", "khue", "khui", "khum", "khun", "khung", "khuoc", "khuon", "khuop", "khuou", "khup", "khuy", "khuyen", "khuyn", "khuyt", "ki", "kia", "kiec", "kiem", "kien", "kiep", "kiet", "kieu", "kim", "kin", "kinh", "kip", "kit",
    "la", "lac", "lai", "lam", "lan", "lang", "lanh", "lao", "lap", "lat", "lau", "lay", "le", "lem", "len", "leo", "let", "li", "lia", "liec", "liem", "lien", "liep", "liet", "lieu", "lim", "lin", "linh", "lip", "lit", "lo", "loa", "loai", "loan", "loang", "loap", "loat", "loc", "loi", "lom", "lon", "long", "lop", "lot", "lu", "lua", "luc", "lui", "lum", "lun", "lung", "luoc", "luoi", "luon", "luop", "luot", "luu", "luyen", "luyt",
    "ma", "mac", "mai", "mam", "man", "mang", "manh", "mao", "map", "mat", "mau", "may", "me", "mem", "men", "meo", "met", "mi", "mia", "miec", "miem", "mien", "miep", "miet", "mieu", "mim", "min", "minh", "mit", "mo", "moa", "moc", "moi", "mom", "mon", "mong", "mop", "mot", "mu", "mua", "muc", "mui", "mum", "mun", "mung", "muoc", "muoi", "muon", "muot", "muu",
    "na", "nac", "nai", "nam", "nan", "nang", "nanh", "nao", "nap", "nat", "nau", "nay", "ne", "nem", "nen", "neo", "net", "ng", "nga", "ngac", "ngai", "ngam", "ngan", "ngang", "nganh", "ngao", "ngap", "ngat", "ngau", "ngay", "nge", "ngem", "ngen", "ngeo", "nget", "nghi", "nghia", "nghiec", "nghiem", "nghien", "nghiep", "nghiet", "nghieu", "nghim", "nghin", "nghinh", "nghit", "ngo", "ngoa", "ngoac", "ngoai", "ngoan", "ngoang", "ngoat", "ngoc", "ngoi", "ngom", "ngon", "ngong", "ngop", "ngot", "ngu", "ngua", "nguc", "ngue", "ngui", "ngum", "ngun", "ngung", "nguoc", "nguoi", "nguon", "nguot", "nguu", "nguy", "nguyen", "nguyt", "nha", "nhac", "nhai", "nham", "nhan", "nhang", "nhanh", "nhao", "nhap", "nhat", "nhau", "nhay", "nhe", "nhem", "nhen", "nheo", "nhet", "nhi", "nhia", "nhiec", "nhiem", "nhien", "nhiep", "nhiet", "nhieu", "nhim", "nhin", "nhinh", "nhit", "nho", "nhoa", "nhoac", "nhoai", "nhoan", "nhoang", "nhoat", "nhoc", "nhoi", "nhom", "nhon", "nhong", "nhop", "nhot", "nhu", "nhua", "nhuc", "nhue", "nhui", "nhum", "nhun", "nhung", "nhuoc", "nhuoi", "nhuon", "nhuot", "nhuu", "nhuy", "nhuyen", "nhuyn", "nhuyt", "ni", "nia", "niec", "niem", "nien", "niep", "niet", "nieu", "nim", "nin", "ninh", "nit", "no", "noa", "noan", "noang", "noat", "noc", "noi", "nom", "non", "nong", "nop", "not", "nu", "nua", "nuc", "nui", "num", "nun", "nung", "nuoc", "nuoi", "nuon", "nuot",
    "oa", "oac", "oai", "oam", "oan", "oang", "oanh", "oap", "oat", "oay", "oc", "oi", "om", "on", "ong", "op", "ot",
    "pa", "pha", "phac", "phai", "pham", "phan", "phang", "phanh", "phao", "phap", "phat", "phau", "phay", "phe", "phem", "phen", "pheo", "phet", "phi", "phia", "phiec", "phiem", "phien", "phiep", "phiet", "phieu", "phim", "phin", "phinh", "phit", "pho", "phoa", "phoan", "phoang", "phoc", "phoi", "phom", "phon", "phong", "phot", "phu", "phua", "phuc", "phui", "phum", "phun", "phung", "phuoc", "phuoi", "phuon", "phuop", "phuot", "phuong", "phuy", "phuyen", "pin",
    "qua", "quac", "quai", "quam", "quan", "quang", "quanh", "quao", "quap", "quat", "quau", "quay", "que", "quem", "quen", "queo", "quet", "qui", "quim", "quin", "quinh", "quit", "quo", "quoc", "quoi", "quom", "quon", "quong", "quot", "quu", "quy", "quyet", "quynh", "quyt",
    "ra", "rac", "rai", "ram", "ran", "rang", "ranh", "rao", "rap", "rat", "rau", "ray", "re", "rem", "ren", "reo", "ret", "ri", "ria", "riec", "riem", "rien", "riep", "riet", "rieu", "rim", "rin", "rinh", "rip", "rit", "ro", "roa", "roac", "roan", "roang", "roat", "roc", "roi", "rom", "ron", "rong", "rop", "rot", "ru", "rua", "ruc", "rui", "rum", "run", "rung", "ruoc", "ruoi", "ruon", "ruot", "ruou", "ruy", "ruyen",
    "sa", "sac", "sai", "sam", "san", "sang", "sanh", "sao", "sap", "sat", "sau", "say", "se", "sem", "sen", "seo", "set", "si", "sia", "siec", "siem", "sien", "siep", "siet", "sieu", "sim", "sin", "sinh", "sip", "sit", "so", "soa", "soac", "soan", "soang", "soat", "soc", "soi", "som", "son", "song", "sop", "sot", "su", "sua", "suc", "sui", "sum", "sun", "sung", "suoc", "suoi", "suon", "suot", "suu", "suy", "xuyen",
    "ta", "tac", "tai", "tam", "tan", "tang", "tanh", "tao", "tap", "tat", "tau", "tay", "te", "tem", "ten", "teo", "tet", "tha", "thac", "thai", "tham", "than", "thang", "thanh", "thao", "thap", "that", "thau", "thay", "the", "them", "then", "theo", "thet", "thi", "thia", "thiec", "thiem", "thien", "thiep", "thiet", "thieu", "thim", "thin", "thinh", "thip", "thit", "tho", "thoa", "thoac", "thoai", "thoan", "thoang", "thoat", "thoc", "thoi", "thom", "thon", "thong", "thop", "thot", "thu", "thua", "thuc", "thue", "thui", "thum", "thun", "thung", "thuoc", "thuoi", "thuon", "thuop", "thuot", "thuong", "thuu", "thuy", "thuyen", "thuyn", "thuyt", "ti", "tia", "tiec", "tiem", "tien", "tiep", "tiet", "tieu", "tim", "tin", "tinh", "tip", "tit", "to", "toa", "toac", "toai", "toan", "toang", "toat", "toc", "toi", "tom", "ton", "tong", "top", "tot", "tra", "trac", "trai", "tram", "tran", "trang", "tranh", "trao", "trap", "trat", "trau", "tray", "tre", "trem", "tren", "treo", "tret", "tri", "tria", "triec", "triem", "trien", "triep", "triet", "trieu", "trim", "trin", "trinh", "trip", "trit", "tro", "troa", "troan", "troang", "troat", "troc", "troi", "trom", "tron", "trong", "trop", "trot", "tru", "trua", "truc", "true", "trui", "trum", "trun", "trung", "truoc", "truoi", "truon", "truot", "truu", "truy", "truyen", "truyn", "truyt", "tu", "tua", "tuc", "tue", "tui", "tum", "tun", "tung", "tuoc", "tuoi", "tuon", "tuot", "tuong", "tuu", "tuy", "tuyen", "tuyn", "tuyt",
    "ua", "uac", "uai", "uan", "uang", "uat", "uc", "ui", "um", "un", "ung", "uoc", "uoi", "uon", "uot", "uou", "up", "ut", "uy", "uyen", "uyet", "uyn", "uyt",
    "va", "vac", "vai", "vam", "van", "vang", "vanh", "vao", "vap", "vat", "vau", "vay", "ve", "vem", "ven", "veo", "vet", "vi", "via", "viec", "viem", "vien", "viep", "viet", "vieu", "vim", "vin", "vinh", "vip", "vit", "vo", "voa", "voan", "voang", "voc", "voi", "vom", "von", "vong", "vop", "vot", "vu", "vua", "vuc", "vui", "vum", "vun", "vung", "vuoc", "vuoi", "vuon", "vuot", "vuong", "vuu", "vuy", "vuyen",
    "xa", "xac", "xai", "xam", "xan", "xang", "xanh", "xao", "xap", "xat", "xau", "xay", "xe", "xem", "xen", "xeo", "xet", "xi", "xia", "xiec", "xiem", "xien", "xiep", "xiet", "xieu", "xim", "xin", "xinh", "xip", "xit", "xo", "xoa", "xoac", "xoai", "xoan", "xoang", "xoat", "xoc", "xoi", "xom", "xon", "xong", "xop", "xot", "xu", "xua", "xuc", "xue", "xui", "xum", "xun", "xung", "xuoc", "xuoi", "xuon", "xuot", "xuong", "xuu", "xuy", "xuyen", "xuyn", "xuyt",
    "y", "yem", "yen", "yet", "yeu"
}

def split_glued_syllables(word: str) -> list:
    """
    Splits glued unaccented Vietnamese syllables using longest-matching Dynamic Programming.
    E.g. 'KYNANGCHONDUNGBIEUDO' -> ['Ky', 'nang', 'chon', 'dung', 'bieu', 'do']
    """
    clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    if not clean or len(clean) < 4:
        return [word]

    n = len(clean)
    # dp[i] = list of words partitioning clean[:i]
    dp = {0: []}
    for i in range(1, n + 1):
        for j in range(max(0, i - 8), i):
            sub = clean[j:i]
            if j in dp and (sub in VN_SYLLABLES or (len(sub) == 1 and sub in "aioeu")):
                candidate = dp[j] + [sub]
                if i not in dp or len(candidate) < len(dp[i]):
                    dp[i] = candidate

    if n in dp and len(dp[n]) > 1:
        # Match case
        res = [w.capitalize() if idx == 0 else w for idx, w in enumerate(dp[n])]
        return res
    return [word]

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

# Test segmentations
test_words = [
    "KYNANGCHONDUNGBIEUDO",
    "Chuyende",
    "Diali",
    "Cocaugiatrisanxuatcongnghiep",
    "Cuacanuocphantheothanhphankinhte",
    "Cuacongnghiepchebienluongthuc"
]

print("--- Syllable Segmentation Test ---")
for tw in test_words:
    tokens = split_glued_syllables(tw)
    print(f"{tw:32} -> {' '.join(tokens)}")
