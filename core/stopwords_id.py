"""
Indonesian text preprocessing: stopword removal + Sastrawi stemming.

Sastrawi is the de-facto Indonesian stemmer (Nazief & Adriani algorithm).
Output is more accurate than the lightweight suffix-stripper.
"""
from __future__ import annotations

import re

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _STEMMER = StemmerFactory().create_stemmer()
    _SASTRAWI_OK = True
except Exception:  # noqa: BLE001
    _STEMMER = None
    _SASTRAWI_OK = False


# Indonesian stopwords (curated set of ~250 most common function words).
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
manakala manapun masa masalah masalahnya masih masihkah
masing-masing maupun melainkan memenuhi mengatakan mengarahnya merasa
mereka merekalah merupakan meski meskipun meyakinkan minta mirip mohon
mulai mungkinkah mungkinlah nah naik namun nanti nataranya nya nyatanya
orang pada padahal padanya pak panitia pasti perlu pernah perwakilan
pergi pergilah pertama pikir waktu ketika saat kala disaat tatkala demi
supaya agar hendak bermaksud mau ingin hendaklah mari ayo silakan
silahkan yuk maupun ataupun ataukah entah manakah bagaikan seakan
seakan-akan seolah seolah-olah rupanya konon katanya semula awalnya
akhirnya kemudian lalu terus terang jelas nyaris hampir kira-kira
kira agaknya mungkin barangkali rasanya agaknya tampaknya kelihatannya
sepertinya seakan-akan hendaknya mestinya seharusnya selayaknya
sebaiknya alangkah betapa betapapun sekalipun senantiasa terus-menerus
kerap-kali
"""

ID_STOPWORDS = frozenset(
    w for w in _ID_STOPWORD_RAW.lower().split() if w and w.isalpha()
)

# Add critical words for academic text
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
    "menurut", "yakin", "pasti", "tentu", "memang", "memang",
    "bahkan", "yaitu", "yakni", "bahwa", "bahwasanya", "biar", "biarpun",
}


def tokenize_id(text: str) -> list[str]:
    """Tokenize Indonesian text. Lowercase, drop punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    return text.split()


def remove_id_stopwords(words: list[str]) -> list[str]:
    return [w for w in words if w not in ID_STOPWORDS]


def stem_id_text(text: str) -> str:
    """Stem a chunk of Indonesian text using Sastrawi."""
    if not _SASTRAWI_OK:
        return text
    return _STEMMER.stem(text)


def stem_id_words(words: list[str]) -> list[str]:
    """Stem a list of words via Sastrawi. Batch for speed."""
    if not _SASTRAWI_OK:
        return words
    joined = " ".join(words)
    stemmed = _STEMMER.stem(joined)
    return stemmed.split()


def preprocess_id(text: str) -> list[str]:
    """Full pipeline: tokenize → stopword removal → Sastrawi stemming."""
    words = tokenize_id(text)
    words = remove_id_stopwords(words)
    words = stem_id_words(words)
    return words
