import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

SYSTEM = (
    "Clasifica la intención del usuario. Responde solo JSON válido con una clave "
    "intent y un valor entre DOMOTICA, MUSICA o GENERAL."
)


def item(text, noise=False, amb=False):
    return {"text": text, "noise": noise, "amb": amb}


DOMOTICA = [
    item("apaga la luz del salón"),
    item("enciende la luz de la cocina"),
    item("prende la del pasillo", amb=True),
    item("abre la persiana del cuarto"),
    item("cierra las persianas del frente"),
    item("sube la cortina del estudio"),
    item("baja la del balcón a la mitad", amb=True),
    item("enciende el ventilador del techo"),
    item("apaga el aire acondicionado"),
    item("pon la calefacción a 22"),
    item("baja el termostato a 19"),
    item("activa la escena noche"),
    item("activa la alarma perimetral"),
    item("cierra la puerta del garaje"),
    item("abre el portón"),
    item("enciende el enchufe de la cafetera"),
    item("apaga el enchufe del televisor"),
    item("cómo está la temperatura del cuarto del bebé ahora mismo para saber si lo tapo"),
    item("dime si la puerta principal está cerrada"),
    item("pasa la aspiradora en la cocina"),
    item("manda el robot al comedor"),
    item("vuelve la aspiradora a la base"),
    item("riega las plantas del patio"),
    item("apaga la luz de la cocina y deja encendida la del pasillo hasta que suba"),
    item("deja solo la lámpara de la esquina", amb=True),
    item("sube la persiana del dormitorio hasta la mitad"),
    item("baja un poco la del estudio", amb=True),
    item("prende luces", amb=True),
    item("modo cine", amb=True),
    item("todo apagado", amb=True),
    item("luz patio", amb=True),
    item("ventilador arriba", amb=True),
    item("abre cortinas", amb=True),
    item("cierra cortina", amb=True),
    item("prende el termo"),
    item("apaga el calentador del baño"),
    item("activa modo fuera de casa"),
    item("desarma la alarma de la entrada antes de abrir la puerta porque ya llegué"),
    item("está abierta la ventana del balcón"),
    item("quiero la sala más fresca sin apagar el ventilador porque todavía hace calor", amb=True),
    item("deja el dormitorio a oscuras pero con luz tenue para no despertar al niño"),
    item("enciende la luz del baño pequeño junto a la cocina que voy medio dormido"),
    item("cuando me vaya apaga todo lo que quede prendido para no gastar de más", amb=True),
    item("pon las luces del comedor en cálido al treinta por ciento para la cena"),
    item("sube un poco más la intensidad de la lámpara del escritorio que sigo leyendo"),
    item("bajame la perziana del living", noise=True),
    item("subi la del fondo", noise=True, amb=True),
    item("prendé las luces de afuera"),
    item("apagá el bentilador del cuarto de los chicos", noise=True),
    item("poné la escena siesta", amb=True),
    item("abrí el garaje"),
    item("serra la puerta de entrada", noise=True),
    item("fijate si hay movimiento en el patio"),
    item("se prendio sola la luz del pasiyo", noise=True),
    item("apaga la lus del baño", noise=True),
    item("habre la puerta del patio que traigo las manos ocupadas", noise=True),
    item("cerrá las cortinas del living"),
    item("prende el bentilador", noise=True),
    item("pon la kalefa en veintiuno", noise=True),
    item("activa la esena relax", noise=True, amb=True),
    item("desactiba la sirena del patio", noise=True),
    item("robot a la cosina", noise=True),
    item("limpia abajo de la mesa del comedor y después vuelve a cargarse solo"),
    item("enciende la luz del pasillo de arriba porque no veo nada al subir"),
    item("baja todas las persianas del frente menos la del despacho de arriba que estoy usando"),
    item("prende las luces del jardín y avísame si se detecta movimiento afuera durante la noche"),
    item("deja el aire en modo seco porque hay mucha humedad en la habitación"),
    item("apaga lo del cuarto", amb=True),
    item("prende lo de afuera", amb=True),
    item("baja la del toldo", amb=True),
    item("pon ambiente lectura", amb=True),
    item("modo dormir", amb=True),
    item("abre la de la sala", amb=True),
    item("cierra la del balcón", amb=True),
    item("deja el cuarto freskito", noise=True, amb=True),
    item("un poco más de luz", amb=True),
    item("menos brillo", amb=True),
    item("revisa si ay alguien en la puerta", noise=True),
    item("desbloquea la cerradura de la entrada para que pase mi hermano que llegó temprano"),
    item("bloquea la puerta principal y confirma si quedó bien cerrada porque me voy a dormir"),
    item("enciende el humidificador"),
    item("apaga el purificador"),
    item("pon el ventilador en velocidad dos"),
    item("enciende solo la luz sobre la mesa"),
    item("apaga la lámpara del buró"),
    item("abre las persianas del salón y la cocina pero deja cerrado el dormitorio"),
    item("quiero la casa lista para dormir con todo apagado salvo el pasillo y el baño", amb=True),
    item("activa la rutina buenos días", amb=True),
    item("desactiva la rutina vacaciones", amb=True),
    item("ay fuga de agua en la labandería", noise=True),
    item("qué marca el sensor de humo de la cocina ahora mismo que siento un olor raro"),
    item("apaga el aire del cuarto de invitados y prende el ventilador de techo"),
    item("deja la sala en veinte grados y las luces tenues para ver una peli"),
    item("kita el aviso del detector", noise=True),
    item("abre la perziana grande", noise=True),
    item("serra el portón porfa", noise=True),
    item("la aspiradora no vaya al cuarto del bebé"),
    item("encendé el foco del pasiyo", noise=True),
    item("apaga el difuzor del dormitorio", noise=True),
    item("pon la escena cena tranquila", amb=True),
]


MUSICA = [
    item("pon jazz suave"),
    item("reproduce salsa clásica"),
    item("pausa la música"),
    item("sigue con lo que estaba sonando"),
    item("reanuda Spotify"),
    item("sube el volumen al cuarenta"),
    item("baja el volumen al veinte"),
    item("pon mi playlist de correr"),
    item("reproduce a Shakira"),
    item("quiero escuchar boleros"),
    item("pon el álbum nuevo de Bad Bunny"),
    item("mezcla algo de rock en español"),
    item("pon música para cocinar"),
    item("quita esta canción"),
    item("vuelve a la anterior"),
    item("pon lo último que escuché ayer"),
    item("activa el modo aleatorio"),
    item("reproduce el álbum en orden"),
    item("repite esta canción"),
    item("guarda esta en favoritos"),
    item("música chill", amb=True),
    item("algo movido", amb=True),
    item("sin canciones explícitas", amb=True),
    item("menos volumen", amb=True),
    item("pon cumbia"),
    item("bachata"),
    item("lo-fi"),
    item("ruido blanco"),
    item("canción romántica", amb=True),
    item("himno de la champions"),
    item("una de Serrat", amb=True),
    item("ponme boleros viejitos para cenar tranquilos sin tanta tristeza ni canciones de ruptura"),
    item("reproduce la lista domingo tranquilo que tengo guardada en Spotify desde el principio"),
    item("abre Spotify y pon mi descubrimiento semanal en el altavoz del salón a volumen medio"),
    item("quiero escuchar el último disco de Karol G completo desde el principio sin mezclar nada"),
    item("pon música para limpiar la casa a buen volumen pero no reguetón ni letras explícitas"),
    item("sáltate esta y busca otra parecida"),
    item("busca versiones acústicas de Juan Luis Guerra para esta tarde mientras cocino en paz"),
    item("pon algo parecido a lo que suena ahora pero más suave y menos triste", amb=True),
    item("bájale un poco porque el bebé está dormido", amb=True),
    item("pon solo artistas mujeres", amb=True),
    item("pone cuarteto cordobés"),
    item("coloca merengue del bueno"),
    item("seguí con la musika", noise=True),
    item("mueve esta al final de la cola"),
    item("reproducí mi mix uno"),
    item("poné Soda Stereo"),
    item("cambiá de artista"),
    item("quiero escuchar a The Weeknd en el salón"),
    item("arranka con algo tranqui", noise=True, amb=True),
    item("pon muzica relajante", noise=True),
    item("reproduse a Arjona", noise=True),
    item("pon espotifai", noise=True),
    item("subele al bolumen", noise=True),
    item("sigiente tema", noise=True),
    item("cuál es esta canción"),
    item("saka este tema de la cola que no pega con lo demás", noise=True),
    item("pon reggeton viejito", noise=True),
    item("buska una playlist de focus", noise=True),
    item("reproduci el albun de Cerati", noise=True),
    item("pon una lista para cenar con amigos que no sea muy intensa ni triste y dure bastante"),
    item("quiero música instrumental para leer sin letra ni sobresaltos esta noche mientras termino trabajo"),
    item("sigue con algo parecido pero un poco más alegre que esto sin cambiar de estilo", amb=True),
    item("pon salsa de los noventa y luego algo de bachata suave para seguir bailando aquí"),
    item("ponme el tema que dice despacito pero no el remix ni la versión en vivo"),
    item("no esa no, la versión en vivo que suena más limpia y no la del estudio"),
    item("mezcla flamenco con pop español a ver qué sale hoy para animar la tarde"),
    item("deja sonando música hasta que me duerma y luego bájala sola para no despertarme"),
    item("pon algo para el asado", amb=True),
    item("algo para estudiar", amb=True),
    item("el volumen nomás", amb=True),
    item("música de fondo", amb=True),
    item("lo mismo pero más calmado", amb=True),
    item("una para bailar", amb=True),
    item("más bajito", amb=True),
    item("adelanta treinta segundos"),
    item("seguí", amb=True),
    item("añade esta canción a mi playlist de carretera"),
    item("pon el top 50 España y luego el top 50 México"),
    item("arranca Spotify en el altavoz de la cocina y sube un poco"),
    item("reproduce en todos los parlantes menos en el cuarto del bebé que ya se durmió"),
    item("quita el modo aleatorio y sigue el álbum desde la tercera canción sin saltarte nada"),
    item("quiero escuchar yaneras", noise=True),
    item("dame marachi", noise=True),
    item("pon trap argentino"),
    item("reprodusí la de ayer", noise=True),
    item("subi el volúmen de la musica", noise=True),
    item("pon la plaslist del gym", noise=True),
    item("siguí con espotifi", noise=True),
    item("bajale a esa musika", noise=True),
    item("activa repetisión", noise=True),
    item("saca el shufle", noise=True),
    item("la que dice corazón partío", amb=True),
    item("una de LuisMi", amb=True),
    item("pon canto gregoriano"),
    item("para la rola"),
    item("quiero un tema de despecho para cantar en la cocina mientras termino de ordenar"),
    item("pon electrónica suave que no tenga drops bruscos porque estoy trabajando desde casa"),
    item("arranca con una versión en directo de Sabina y luego sigue parecido sin irte al rock"),
    item("busca una playlist de reggaetón viejo para la fiesta del sábado que tenga clásicos"),
]


GENERAL = [
    item("qué tiempo hace mañana"),
    item("cuánto es 17 por 23"),
    item("qué hora es"),
    item("cuéntame un chiste"),
    item("quién eres"),
    item("buenas noches"),
    item("buenos días"),
    item("gracias"),
    item("qué puedes hacer"),
    item("recuérdame llamar a mamá"),
    item("cuál es la capital de Chile"),
    item("cuánto falta para navidad"),
    item("va a llover hoy"),
    item("dame una receta de lentejas"),
    item("cómo se dice umbrella en español"),
    item("explícamelo fácil"),
    item("qué significa resiliencia"),
    item("cuánto son cien dólares en euros"),
    item("dime una noticia de tecnología"),
    item("abre YouTube", amb=True),
    item("qué día cae el feriado"),
    item("agenda una alarma para las seis de la mañana porque mañana tengo cita temprano"),
    item("pon un temporizador de diez minutos"),
    item("llama a Ana"),
    item("manda un mensaje a Carlos que llego tarde"),
    item("silencio", amb=True),
    item("ayuda", amb=True),
    item("repite lo último", amb=True),
    item("otra vez", amb=True),
    item("no entendiste", amb=True),
    item("cuál es mi agenda hoy"),
    item("cómo está el tráfico al centro"),
    item("cuántas calorías tiene una arepa"),
    item("quién ganó el mundial del 2010"),
    item("cuánto mide el monte Everest"),
    item("qué significa esa luz roja del router"),
    item("me escuchas", amb=True),
    item("estás ahí", amb=True),
    item("necesito ideas para cenar rápido con lo que tengo en la nevera y sin salir"),
    item("recomiéndame una serie para ver esta noche que no sea muy larga ni muy densa"),
    item("si salgo ahora necesito paraguas o mejor me llevo chaqueta por si refresca luego"),
    item("cuánto tardaría en llegar al aeropuerto un viernes por la tarde saliendo desde casa"),
    item("explícame por qué el cielo cambia de color al atardecer como si fuera un niño"),
    item("acuérdame sacar la basura mañana después del desayuno sin falta porque siempre se me pasa"),
    item("dame tres opciones de desayuno con huevo y aguacate para mañana que no tarden mucho"),
    item("cuéntame algo interesante de Surinam que no sepa casi nadie y sea fácil de recordar"),
    item("cómo va el dólar hoy y si subió respecto a ayer en mi país"),
    item("cuánta batería le queda al móvil"),
    item("busca dónde queda la farmacia de guardia más cercana abierta ahora y cómo llego"),
    item("ke ora es", noise=True),
    item("cuanto e 19 por 8", noise=True),
    item("va a yobé mañana", noise=True),
    item("esplicame que es la fotosintesis", noise=True),
    item("cuentame una adivinansa facil", noise=True),
    item("kien gano el partido de ayer", noise=True),
    item("recordame pagar la luz el viernes", noise=True),
    item("pon un timer de dies minutos", noise=True),
    item("yamá a mamá", noise=True),
    item("mandale un mensage a Lau", noise=True),
    item("qué hubo", amb=True),
    item("to bien", noise=True, amb=True),
    item("hola", amb=True),
    item("eh", amb=True),
    item("y entonces", amb=True),
    item("hace frío afuera", amb=True),
    item("cuánto falta", amb=True),
    item("cómo así", amb=True),
    item("qué dijiste", amb=True),
    item("me ayudas", amb=True),
    item("dime algo", amb=True),
    item("qué pasó hoy en el país y si afecta algo para mañana en el trabajo"),
    item("estaba pensando si mañana hará calor porque tengo que salir temprano con los niños"),
    item("necesito que me expliques como si tuviera diez años qué es la inflación en pocas palabras"),
    item("si pongo arroz ahora, cuánto tiempo debería dejarlo para que no se pase"),
    item("recuérdame dentro de media hora revisar el horno porque siempre se me olvida"),
    item("cuál es la diferencia entre un asteroide, un meteoro y un meteorito en pocas palabras"),
    item("hoy me siento raro, qué puedo hacer para relajarme un poco sin salir de casa"),
    item("si viajo a Madrid el mes que viene, qué ropa me conviene llevar"),
    item("oye tengo sueño pero todavía me falta trabajar, algún consejo para aguantar un rato"),
    item("pon una alarma", amb=True),
    item("sube las noticias", amb=True),
    item("baja la voz", amb=True),
    item("abre el clima", amb=True),
    item("quiero escuchar las noticias", amb=True),
    item("activa no molestar", amb=True),
    item("prende yutub", noise=True),
    item("lee mis notificaciones"),
    item("cuánto ta marcando hoy", noise=True),
    item("oye q hora e", noise=True),
    item("como se escrive recivir", noise=True),
    item("cuanto falta pal viernes", noise=True),
    item("desime un chiste cortito", noise=True),
    item("no me akuerdo de la capital de peru", noise=True),
    item("pon un recordatorio pa mañana", noise=True),
    item("cual es el pronostiko del finde", noise=True),
    item("quiero saber si mañana necesito abrigo para salir con los niños temprano al cole"),
    item("explícame por qué el pan se pone duro cuando lo dejo afuera toda la noche"),
    item("si tengo que hacer una tortilla para seis personas, cuántos huevos uso más o menos"),
    item("cuál fue la última peli de Pixar"),
    item("qué significa cpu"),
]


ALL_SETS = {
    "DOMOTICA": DOMOTICA,
    "MUSICA": MUSICA,
    "GENERAL": GENERAL,
}


def strip_accents(text):
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def normalize_text(text):
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    return len(re.findall(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ-]+\b", text, flags=re.UNICODE))


def build_records():
    for label, rows in ALL_SETS.items():
        assert len(rows) == 100, (label, len(rows))

    seen = {}
    for label, rows in ALL_SETS.items():
        for row in rows:
            key = normalize_text(row["text"])
            assert key not in seen, f"Exact normalized duplicate: {row['text']} == {seen[key]}"
            seen[key] = row["text"]

    records = []
    split_counts = defaultdict(Counter)
    for label, rows in ALL_SETS.items():
        for idx, row in enumerate(rows):
            split = "valid" if idx % 5 == 0 else "train"
            split_counts[split][label] += 1
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["text"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"intent": label}, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                ]
            }
            records.append((split, label, row, record))

    return records, split_counts


def suspicious_pairs():
    rows = [
        (label, row["text"], normalize_text(row["text"]))
        for label, values in ALL_SETS.items()
        for row in values
    ]
    pairs = []
    for i, (label_i, text_i, norm_i) in enumerate(rows):
        for label_j, text_j, norm_j in rows[i + 1 :]:
            ratio = SequenceMatcher(None, norm_i, norm_j).ratio()
            if ratio >= 0.84:
                pairs.append((ratio, label_i, text_i, label_j, text_j))
    pairs.sort(reverse=True)
    return pairs[:20]


def main():
    records, split_counts = build_records()
    all_rows = [row for _, _, row, _ in records]
    totals = Counter(label for _, label, _, _ in records)
    noise_count = sum(1 for row in all_rows if row["noise"])
    amb_count = sum(1 for row in all_rows if row["amb"])
    short_count = sum(1 for row in all_rows if 1 <= word_count(row["text"]) <= 3)
    long_count = sum(1 for row in all_rows if word_count(row["text"]) > 12)
    for label in ALL_SETS:
        assert totals[label] == 100, (label, totals[label])
        assert split_counts["train"][label] == 80, (label, split_counts["train"][label])
        assert split_counts["valid"][label] == 20, (label, split_counts["valid"][label])
    assert noise_count >= 60, noise_count
    assert amb_count >= 40, amb_count
    assert short_count >= 50, short_count
    assert long_count >= 50, long_count

    print(
        json.dumps(
            {
                "totals": totals,
                "splits": {split: dict(counts) for split, counts in split_counts.items()},
                "noise": noise_count,
                "ambiguous": amb_count,
                "short_1_3": short_count,
                "long_gt_12": long_count,
                "train_lines": sum(1 for split, _, _, _ in records if split == "train"),
                "valid_lines": sum(1 for split, _, _, _ in records if split == "valid"),
            },
            ensure_ascii=False,
            default=dict,
        )
    )
    print("SUSPICIOUS_PAIRS")
    for ratio, label_i, text_i, label_j, text_j in suspicious_pairs():
        print(f"{ratio:.2f} | {label_i} | {text_i} || {label_j} | {text_j}")

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    for split in ("train", "valid"):
        lines = [
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            for current_split, _, _, record in records
            if current_split == split
        ]
        (data_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
