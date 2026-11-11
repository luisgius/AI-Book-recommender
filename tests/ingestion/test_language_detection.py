"""Tests for the dependency-free language detector used by the catalog builder."""

import pytest

from app.ingestion.language_detection import detect_language

SAMPLES = {
    "en": "Pride and Prejudice is a novel by Jane Austen. It follows the character development of "
          "Elizabeth Bennet, the protagonist of the book, who learns about the repercussions of hasty "
          "judgments and comes to appreciate the difference between superficial goodness and actual goodness.",
    "es": "Cien años de soledad es una novela del escritor colombiano Gabriel García Márquez, ganador del "
          "Premio Nobel de Literatura en 1982. Es considerada una obra maestra de la literatura "
          "hispanoamericana y universal, así como una de las obras más traducidas y leídas en español.",
    "fr": "Les Misérables est un roman de Victor Hugo paru en 1862. Il a donné lieu à de nombreuses "
          "adaptations, au cinéma et sur de nombreux autres supports. Dans ce roman, l'auteur décrit la vie "
          "de misérables dans Paris et la France provinciale du dix-neuvième siècle.",
    "de": "Die Verwandlung ist eine Erzählung von Franz Kafka, die im Jahr 1912 entstand und 1915 "
          "erschien. Sie handelt von Gregor Samsa, der eines Morgens als Ungeziefer erwacht und von seiner "
          "Familie zunehmend abgelehnt wird, bis er schließlich stirbt.",
    "it": "I promessi sposi è un celebre romanzo storico di Alessandro Manzoni, ritenuto il più famoso e il "
          "più letto tra quelli scritti in lingua italiana. Il romanzo è ambientato in Lombardia tra il 1628 e "
          "il 1630, durante il dominio spagnolo.",
    "pt": "Dom Casmurro é um romance escrito por Machado de Assis, publicado em 1899. O livro é narrado em "
          "primeira pessoa por Bento Santiago, que conta a sua vida desde a juventude e o seu casamento com "
          "Capitu, a quem acusa de traição com o seu melhor amigo.",
    "ca": "Tirant lo Blanc és una novel·la cavalleresca escrita per Joanot Martorell i acabada per Martí Joan "
          "de Galba. És una de les obres més importants de la literatura catalana i va ser publicada per "
          "primera vegada a València l'any 1490.",
    "nl": "Max Havelaar, of de koffi-veilingen der Nederlandsche Handel-Maatschappy is een roman van "
          "Multatuli die in 1860 werd gepubliceerd. Het boek wordt gezien als een van de belangrijkste werken "
          "van de Nederlandse literatuur en het heeft een grote invloed gehad op het koloniale beleid.",
    "ru": "Война и мир — роман-эпопея Льва Николаевича Толстого, описывающий русское общество в эпоху войн "
          "против Наполеона в 1805—1812 годах. Эпилог романа доводит повествование до 1820 года.",
    "zh": "紅樓夢是中國古典長篇章回小說，被列為中國古典四大名著之首。小說以賈、史、王、薛四大家族的興衰為背景，"
          "以賈寶玉、林黛玉、薛寶釵的愛情婚姻悲劇為主線，描寫了封建社會的種種面貌。",
    "ja": "『吾輩は猫である』は、夏目漱石の長編小説であり、処女小説である。中学校の英語教師である珍野苦沙弥の家に"
          "飼われている猫である「吾輩」の視点から、珍野一家や、そこに集う人間たちの姿を風刺的に描いている。",
    "ar": "ألف ليلة وليلة هو كتاب يتضمن مجموعة من القصص التي رويت في المنطقة الممتدة من الصين وحتى المغرب، "
          "وتدور حول ملك يدعى شهريار وزوجته شهرزاد التي تحكي له قصة كل ليلة حتى لا يقتلها.",
}


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_detects_each_configured_language(language):
    assert detect_language(SAMPLES[language]) == language


@pytest.mark.parametrize("text", [None, "", "Hamlet", "1984 1985 1986", "ISBN 978-84-376-0494-7"])
def test_returns_none_when_evidence_is_too_weak(text):
    assert detect_language(text) is None


def test_ukrainian_and_persian_are_told_apart_from_russian_and_arabic():
    assert detect_language("Кобзар — збірка поетичних творів Тараса Шевченка, яка вперше вийшла друком у "
                           "Санкт-Петербурзі в 1840 році. Це одна з найважливіших книг української "
                           "літератури.") == "uk"
    assert detect_language("شاهنامه اثر حکیم ابوالقاسم فردوسی توسی، حماسه‌ای منظوم، بر حسب دست‌نوشته‌های "
                           "موجود دربرگیرنده نزدیک به ۵۰٬۰۰۰ تا ۶۱٬۰۰۰ بیت است و یکی از بزرگ‌ترین و "
                           "برجسته‌ترین سروده‌های حماسی جهان است که سرایش آن دست‌آورد دست‌کم سی سال کار پیوسته") == "fa"
