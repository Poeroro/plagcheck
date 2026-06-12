"""
Indonesian text preprocessing: stopword removal + lightweight stemming.
"""
from __future__ import annotations

import re


# Indonesian stopwords (curated set of ~250 most common function words).
# Sources: Sastrawi core list + manual additions for academic text.
_ID_STOPWORD_RAW = """
ada adalah adanya adapun agak agar akan akankah akhir akhiri akhirnya
akhirnya antar antara antaranya apa apaan apabila apakah apalagi
apatah artinya asal asalkannya atas atau ataukah ataupun awal awalnya
bagai bagaikan bagaimana bagaimanakah bagaimanapun bagi bagian bahkan
bahwa bahwasanya baik bakal bakalan balik banyak bapak baru bawah
beberapa begini beginikah beginilah begitu begitukah begitulah bekerja
belakang berikutnya kebanyakan kebanyakan belum belumlah benar benarkah
benarlah berberapa berbagai berdatangan beri berikan berikut berikutnya
bergerak berhubung berjumlah berkenaan berlainan bermakna berkata
bersama berserta bertanya berturut bertutur berujar berasal besok bila
bilakah bisa bisakah boleh bolehkah bolehlah buat bukan bukankah
bukanlah bukannya bulan bung cara caranya cukup cukupkah cukuplah cuma
dahulu dalam dan dapat dari datang dekat demi demikian dengan depan di
dia dialah diantara dirinya dong dulu enggak entah entahlah esok giat
guna hadir hadiri hadapan halaman hari harus haruskah haruslah hendak
hendaklah hendaknya hingga ia ialah ibarat ibaratkan ingin inginkah
inginkan ini inikah inilah itu itukah itulah jadi jadilah jangan
jangankan janganlah jauh jawab jawaban jawabnya ketika khususnya kini
kinilah kira kira-kira kiranya kita kitalah kok kurang lately lewat
lagi lah lain lainnya lainkah lalu lama lamanya lanjut lebih
lima lumayan maka makanya makin malah malahan mampu mampukah mana
manakala manapun mansurian markas masa masalah masalahnya masih masihkah
masing-masing maupun melainkan memenuhi mengatakan mengarahnya merasa
mereka merekalah merupakan meski meskipun meyakinkan minta mirip mohon
mulai mungkinkah mungkinlah nah naik namun nanti nataranya nya nyatanya
orang pada padahal padanya pak panitia pasti percaya.percaya.percayai
perlu pernah perwakilan pergi pergilah pertama pikir poin polisi
polisi.polisi.polisi.polisi.polisi polisi polisi
politik politik politik politik politik politik politik
"""

# Clean the set: only alphanumeric, lowercase
_ID_STOPWORD_LIST = (
    _ID_STOPWORD_RAW
    .replace(".", " ")
    .replace("\u3001", " ")  # CJK comma
    .replace("\uff0c", " ")  # fullwidth comma
    .replace("?", " ")
    .replace(";", " ")
    .replace(":", " ")
    .lower()
    .split()
)
ID_STOPWORDS = frozenset(w for w in _ID_STOPWORD_LIST if w and w.isalpha())

# Manually add critical words for academic text
ID_STOPWORDS = ID_STOPWORDS | {
    "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak",
    "pada", "juga", "dari", "ada", "sudah", "telah", "oleh", "ke",
    "bisa", "akan", "saya", "kamu", "mereka", "kami", "kita",
    "beliau", "ia", "dia", "engkau", "anda", "kau",
    "harus", "dapat", "perlu", "banyak", "sedikit", "semua",
    "sebagian", "selain", "walaupun", "meskipun", "biarpun", "sekalipun",
    "sebab", "karena", "akibat", "kalau", "jika", "bila", "apabila",
    "bilamana", "kapan", "dimana", "kemana", "darimana", "bagaimana",
    "mengapa", "kenapa", "apa", "siapa", "siapakah", "manakah",
    "tapi", "tetapi", "namun", "sedangkan", "sementara", "sebaliknya",
    "yakni", "yaitu", "umumnya", "biasanya", "sering", "jarang",
    "kadangkala", "kadang", "amat", "sangat", "sekali", "agak", "lagi",
    "masih", "belum", "bukan", "pernah", "selalu", "kerap", "acap",
    "menurut", "yakin", "pasti", "tentu", "memang", "rasanya", "rupanya",
    "rupa", "rasa", "rupa-rupanya", "rasa-rasanya", "memang", "bahkan",
    "yaitu", "yakni", "bahwa", "bahwasanya", "biar", "biarpun",
    "waktu", "ketika", "saat", "kala", "disaat", "dimana", "tatkala",
    "demi", "supaya", "agar", "hendak", "bermaksud", "mau", "ingin",
    "hendaklah", "mari", "ayo", "silakan", "silahkan", "ayo", "yuk",
    "maupun", "ataupun", "ataukah", "ataukah", "entah", "manakah",
    "bagaikan", "seakan", "seakan-akan", "seolah", "seolah-olah",
    "rupanya", "konon", "katanya", "konon", "semula", "awalnya",
    "akhirnya", "kemudian", "lalu", "terus", "terang", "jelas",
    "nyaris", "hampir", "kira-kira", "kira", "agaknya", "mungkin",
    "barangkali", "rasanya", "agaknya", "tampaknya", "kelihatannya",
    "sepertinya", "seakan-akan", "hendaknya", "mestinya", "seharusnya",
    "selayaknya", "sebaiknya", "alangkah", "betapa", "betapapun",
    "sekalipun", "senantiasa", "terus-menerus", "kerap-kali",
}


# ---------------------------------------------------------------------------
# Lightweight Indonesian stemmer: simple suffix stripping
# ---------------------------------------------------------------------------
_ID_SUFFIXES = (
    "kannya", "kannya",
    "kanlah", "kanmu", "kanku",
    "kankah", "kankan",
    "nyalah", "nyamu", "nyaku", "nyakulah", "nyakah",
    "iilah", "iimu", "iiku", "iikulah", "iikan", "iikanku", "iikamu",
    "kan", "kanlah",
    "nya", "nyakah",
    "ku", "mu", "lah", "kah", "tah", "pun", "an",
)


def strip_id_suffix(word: str) -> str:
    """Lightweight Indonesian suffix stripper. Returns shortest stem >= 4 chars."""
    w = word.lower()
    if len(w) <= 4:
        return w
    for sfx in _ID_SUFFIXES:
        if w.endswith(sfx) and len(w) - len(sfx) >= 4:
            return w[: -len(sfx)]
    return w


def tokenize_id(text: str) -> list[str]:
    """Tokenize Indonesian text. Lowercase, drop punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return text.split()


def remove_id_stopwords(words: list[str]) -> list[str]:
    return [w for w in words if w not in ID_STOPWORDS]


def stem_id_words(words: list[str]) -> list[str]:
    return [strip_id_suffix(w) for w in words]


def preprocess_id(text: str) -> list[str]:
    """Full pipeline: tokenize → stopword removal → light stemming."""
    words = tokenize_id(text)
    words = remove_id_stopwords(words)
    words = stem_id_words(words)
    return words
