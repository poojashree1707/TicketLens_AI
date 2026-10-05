import pytesseract
import cv2
import re
import numpy as np
from PIL import Image

if __import__("os").name == "nt":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if __import__("os").path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def normalize(text):
    text = clean_text(text)
    return re.sub(r"[^A-Za-z0-9@:/.,%+\-#() ]", "", text)


def preprocess_image(image):
    img = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    scale = 1.5

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        10,
        7,
        21
    )

    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        img,
        gray,
        threshold
    ]


def ocr_data(image):
    best = None
    best_score = -1

    for processed in preprocess_image(image):

        for psm in [6, 11, 12, 3]:

            data = pytesseract.image_to_data(
                processed,
                output_type=pytesseract.Output.DICT,
                config=f"--oem 3 --psm {psm}"
            )

            words = []
            confidence = []

            for i, text in enumerate(data["text"]):

                text = clean_text(text)

                if text:

                    try:
                        conf = float(data["conf"][i])
                    except:
                        conf = 0

                    words.append({
                        "text": text,
                        "left": int(data["left"][i]),
                        "top": int(data["top"][i]),
                        "width": int(data["width"][i]),
                        "height": int(data["height"][i]),
                        "right": int(data["left"][i]) + int(data["width"][i]),
                        "bottom": int(data["top"][i]) + int(data["height"][i]),
                        "conf": conf,
                        "block": data["block_num"][i],
                        "par": data["par_num"][i],
                        "line": data["line_num"][i]
                    })

                    confidence.append(max(conf, 0))

            score = (
                sum(confidence) / len(confidence)
                if confidence
                else 0
            )

            if len(words) > 5:
                score += min(len(words), 100) * 0.05

            if score > best_score:
                best_score = score
                best = words

    return best or []


def group_lines(words):
    groups = {}

    for word in words:

        key = (
            word["block"],
            word["par"],
            word["line"]
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(word)

    lines = []

    for items in groups.values():

        items = sorted(
            items,
            key=lambda x: x["left"]
        )

        text = " ".join(
            x["text"]
            for x in items
        )

        text = clean_text(text)

        if not text:
            continue

        lines.append({
            "text": text,
            "words": items,
            "left": min(x["left"] for x in items),
            "top": min(x["top"] for x in items),
            "right": max(x["right"] for x in items),
            "bottom": max(x["bottom"] for x in items),
            "center_y": (
                min(x["top"] for x in items)
                + max(x["bottom"] for x in items)
            ) / 2
        })

    lines.sort(
        key=lambda x: (x["top"], x["left"])
    )

    return lines


def all_text_from_lines(lines):
    return "\n".join(
        line["text"]
        for line in lines
    )


def find_value(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(1).strip(" :|-")

            if value:
                return value

    return "Not detected"


def detect_transport_type(text):

    upper = text.upper()

    if any(
        x in upper
        for x in [
            "FLIGHT",
            "AIRLINE",
            "BOARDING PASS",
            "GATE",
            "PNR"
        ]
    ):

        if any(
            x in upper
            for x in [
                "FLIGHT",
                "AIRLINE",
                "BOARDING PASS"
            ]
        ):
            return "Flight"

    if any(
        x in upper
        for x in [
            "TRAIN",
            "RAILWAY",
            "RAIL",
            "COACH",
            "BERTH",
            "PLATFORM"
        ]
    ):
        return "Train"

    if any(
        x in upper
        for x in [
            "BUS",
            "BUS OPERATOR",
            "BUS SERVICE",
            "BUS TICKET",
            "SEAT NO"
        ]
    ):
        return "Bus"

    if any(
        x in upper
        for x in [
            "FERRY",
            "SHIP",
            "VESSEL",
            "CRUISE",
            "PORT"
        ]
    ):
        return "Ship / Ferry"

    if any(
        x in upper
        for x in [
            "TAXI",
            "CAB",
            "UBER",
            "OLA",
            "RIDE"
        ]
    ):
        return "Taxi / Cab"

    return "Other Transport"


def extract_basic_information(text):

    result = {}

    result["Transport Type"] = detect_transport_type(text)

    result["Ticket Number"] = find_value(
        text,
        [
            r"(?:ticket\s*(?:number|no|id)|ticket\s*#|ticket\s*ref(?:erence)?)\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_-]{3,})",
            r"(?:e[- ]?ticket|eticket|booking\s*(?:number|no|id))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/_-]{3,})"
        ]
    )

    result["PNR"] = find_value(
        text,
        [
            r"(?:PNR|PNR\s*No|Booking\s*Reference|Booking\s*ID|Reservation\s*(?:No|Number|ID)|Record\s*Locator)\s*[:#-]?\s*([A-Z0-9-]{4,})"
        ]
    )

    result["Journey Date"] = find_value(
        text,
        [
            r"(?:journey\s*date|travel\s*date|date\s*of\s*journey|departure\s*date|date\s*of\s*travel|date)\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"\b([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b",
            r"\b([0-9]{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b"
        ]
    )

    result["Journey Time"] = find_value(
        text,
        [
            r"(?:departure\s*time|departure|travel\s*time|journey\s*time|boarding\s*time|scheduled\s*time)\s*[:\-]?\s*([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM)?)",
            r"\b([0-9]{1,2}:[0-9]{2}\s*(?:AM|PM))\b",
            r"\b([0-9]{1,2}:[0-9]{2})\b"
        ]
    )

    result["Ticket Price"] = find_value(
        text,
        [
            r"(?:ticket\s*price|total\s*fare|total\s*amount|total\s*price|fare|amount\s*paid|amount|price|grand\s*total)\s*[:\-]?\s*(?:Rs\.?|INR|₹|USD|EUR|GBP|\$|€|£)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"
        ]
    )

    result["Boarding Point"] = find_value(
        text,
        [
            r"(?:boarding\s*point|boarding\s*station|boarding\s*stop|pickup\s*point|pickup|boarding|departure\s*station|origin)\s*[:\-]?\s*([^\n]+)"
        ]
    )

    result["Dropping Point"] = find_value(
        text,
        [
            r"(?:dropping\s*point|drop\s*point|drop\s*stop|drop|destination|arrival\s*station|destination\s*station)\s*[:\-]?\s*([^\n]+)"
        ]
    )

    result["From"] = find_value(
        text,
        [
            r"(?:from|origin|source|depart(?:ure)?\s*(?:from|location)?)\s*[:\-]?\s*([A-Za-z][A-Za-z .,'()&/-]{1,60}?)(?=\s+(?:to|destination|arrival|depart|$))",
            r"\bfrom\s+([A-Za-z][A-Za-z .,'()&/-]{1,60})\s+to\s+[A-Za-z]"
        ]
    )

    result["To"] = find_value(
        text,
        [
            r"(?:to|destination|arrival\s*(?:at|station)?|going\s*to)\s*[:\-]?\s*([A-Za-z][A-Za-z .,'()&/-]{1,60})"
        ]
    )

    result["Operator / Travel Company"] = find_value(
        text,
        [
            r"(?:operator|bus\s*operator|train\s*operator|airline|carrier|travel\s*operator|travels|service\s*provider|transport\s*operator|company)\s*[:\-]?\s*([^\n]+)"
        ]
    )

    result["Bus / Service"] = find_value(
        text,
        [
            r"(?:bus\s*name|bus\s*number|bus\s*no|train\s*name|train\s*number|train\s*no|flight\s*(?:number|no)|flight\s*name|service\s*name|service\s*number|service|vehicle\s*(?:number|no|name)|route\s*number)\s*[:#-]?\s*([^\n]+)"
        ]
    )

    result["Seat / Coach"] = find_value(
        text,
        [
            r"(?:seat\s*(?:number|no|#)?|seat|berth|coach|cabin|compartment)\s*[:#-]?\s*([A-Z0-9-]{1,15})"
        ]
    )

    result["Platform / Gate"] = find_value(
        text,
        [
            r"(?:platform|gate)\s*(?:number|no|#)?\s*[:#-]?\s*([A-Z0-9-]{1,10})"
        ]
    )

    result["Booking Name"] = find_value(
        text,
        [
            r"(?:passenger\s*name|traveller\s*name|traveler\s*name|customer\s*name|guest\s*name|booked\s*by|name)\s*[:\-]?\s*([A-Za-z][A-Za-z .'-]{2,60})",
            r"(?:hello|hey|hi)\s+([A-Za-z][A-Za-z .]{2,})[,!]"
        ]
    )

    return result


def is_gender(text):

    value = re.sub(
        r"[^A-Za-z]",
        "",
        text.upper()
    )

    return value in {
        "MALE",
        "FEMALE",
        "M",
        "F",
        "MAN",
        "WOMAN",
        "BOY",
        "GIRL"
    }


def extract_gender(text):

    match = re.search(
        r"\b(FEMALE|MALE|WOMAN|MAN|GIRL|BOY|F|M)\b",
        text,
        re.IGNORECASE
    )

    if not match:
        return "Not detected"

    value = match.group(1).upper()

    if value in {
        "FEMALE",
        "WOMAN",
        "GIRL",
        "F"
    }:
        return "Female"

    if value in {
        "MALE",
        "MAN",
        "BOY",
        "M"
    }:
        return "Male"

    return "Not detected"


def extract_age(text):

    patterns = [
        r"\b(\d{1,3})\s*(?:years?|yrs?|y/o)\b",
        r"\b(?:age)\s*[:\-]?\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*,?\s*(?:years?|yrs?)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            age = int(match.group(1))

            if 0 <= age <= 120:
                return str(age)

    match = re.search(
        r"\b(\d{1,3})\s*%?\s*,?\s*(?:MALE|FEMALE|M|F)\b",
        text,
        re.IGNORECASE
    )

    if match:

        age = int(match.group(1))

        if 0 <= age <= 120:
            return str(age)

    return "Not detected"


def is_section_heading(text):

    value = re.sub(
        r"[^A-Za-z ]",
        " ",
        text
    ).upper()

    value = clean_text(value)

    headings = [
        "PASSENGER DETAILS",
        "PASSENGER INFORMATION",
        "PASSENGER LIST",
        "TRAVELLER DETAILS",
        "TRAVELER DETAILS",
        "TRAVELLER INFORMATION",
        "TRAVELER INFORMATION",
        "CUSTOMER DETAILS",
        "GUEST DETAILS",
        "PERSON DETAILS",
        "BOOKING DETAILS"
    ]

    return any(
        heading in value
        for heading in headings
    )


def passenger_section_indexes(lines):

    start = None

    for i, line in enumerate(lines):

        if is_section_heading(line["text"]):
            start = i + 1
            break

    if start is None:

        for i, line in enumerate(lines):

            text = line["text"].upper()

            if (
                "PASSENGER" in text
                or "TRAVELLER" in text
                or "TRAVELER" in text
                or "GUEST" in text
            ):
                start = i + 1
                break

    if start is None:
        return 0, len(lines)

    end = len(lines)

    stop_words = [
        "TERMS AND CONDITIONS",
        "TERMS & CONDITIONS",
        "IMPORTANT INFORMATION",
        "CANCELLATION POLICY",
        "CONTACT US",
        "THANK YOU",
        "NOTE:"
    ]

    for i in range(start, len(lines)):

        upper = lines[i]["text"].upper()

        if any(
            word in upper
            for word in stop_words
        ):
            end = i
            break

    return start, end


def name_from_line(text):

    value = re.sub(
        r"\b(?:MALE|FEMALE|M|F|YEARS?|YRS?|Y/O|AGE)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    value = re.sub(
        r"\b\d{1,3}\s*%?\b",
        " ",
        value
    )

    value = re.sub(
        r"[^A-Za-z .'-]",
        " ",
        value
    )

    value = clean_text(value)

    bad = {
        "PASSENGER",
        "DETAILS",
        "SEAT",
        "NO",
        "AGE",
        "GENDER",
        "NAME",
        "TRAVELLER",
        "TRAVELER",
        "CUSTOMER",
        "PERSON"
    }

    words = value.split()

    words = [
        word
        for word in words
        if word.upper() not in bad
        and len(word) > 1
    ]

    if not words:
        return "Not detected"

    return " ".join(words[:6])


def seat_from_words(words, image_width):

    candidates = []

    for word in words:

        text = word["text"].strip()

        if re.fullmatch(
            r"\d{1,4}",
            text
        ):

            number = int(text)

            if 1 <= number <= 9999:

                if word["left"] > image_width * 0.45:
                    candidates.append(
                        (word["top"], number)
                    )

    if candidates:

        candidates.sort(
            key=lambda x: x[0]
        )

        return str(candidates[0][1])

    return "Not detected"


def get_age_gender_lines(lines):

    found = []

    for i, line in enumerate(lines):

        text = line["text"]

        gender = extract_gender(text)

        if gender == "Not detected":
            continue

        age = extract_age(text)

        found.append({
            "index": i,
            "gender": gender,
            "age": age,
            "text": text
        })

    return found


def extract_passengers(lines, image_width):

    start, end = passenger_section_indexes(lines)

    section = lines[start:end]

    gender_lines = get_age_gender_lines(section)

    passengers = []

    used_names = set()

    for item in gender_lines:

        index = item["index"]

        current = section[index]

        name = "Not detected"

        if index > 0:

            previous = section[index - 1]

            if previous["top"] < current["top"]:

                candidate = name_from_line(
                    previous["text"]
                )

                if candidate != "Not detected":
                    name = candidate

        if name == "Not detected":

            candidate = name_from_line(
                current["text"]
            )

            if candidate != "Not detected":
                name = candidate

        if name in used_names and index > 1:

            candidate = name_from_line(
                section[index - 2]["text"]
            )

            if candidate != "Not detected":
                name = candidate

        seat = "Not detected"

        for j in range(
            max(0, index - 2),
            min(len(section), index + 2)
        ):

            found_seat = seat_from_words(
                section[j]["words"],
                image_width
            )

            if found_seat != "Not detected":
                seat = found_seat
                break

        if name == "Not detected":
            continue

        if len(name) < 2:
            continue

        if name.upper() in {
            "PASSENGER",
            "DETAILS",
            "SEAT",
            "GENDER",
            "AGE"
        }:
            continue

        passengers.append({
            "Name": name,
            "Age": item["age"],
            "Gender": item["gender"],
            "Seat No": seat
        })

        used_names.add(name)

    unique = []

    seen = set()

    for passenger in passengers:

        key = (
            passenger["Name"].upper(),
            passenger["Gender"],
            passenger["Seat No"]
        )

        if key not in seen:

            seen.add(key)

            unique.append(passenger)

    return unique


def fallback_passengers(lines, image_width):

    passengers = []

    for i, line in enumerate(lines):

        gender = extract_gender(
            line["text"]
        )

        if gender == "Not detected":
            continue

        age = extract_age(
            line["text"]
        )

        possible_names = []

        if i > 0:
            possible_names.append(
                lines[i - 1]
            )

        if i > 1:
            possible_names.append(
                lines[i - 2]
            )

        name = "Not detected"

        for candidate_line in possible_names:

            candidate = name_from_line(
                candidate_line["text"]
            )

            if candidate != "Not detected":

                name = candidate
                break

        if name == "Not detected":

            candidate = name_from_line(
                line["text"]
            )

            if candidate != "Not detected":
                name = candidate

        seat = seat_from_words(
            line["words"],
            image_width
        )

        if seat == "Not detected" and i > 0:

            seat = seat_from_words(
                lines[i - 1]["words"],
                image_width
            )

        if name != "Not detected":

            passengers.append({
                "Name": name,
                "Age": age,
                "Gender": gender,
                "Seat No": seat
            })

    unique = []

    seen = set()

    for passenger in passengers:

        key = (
            passenger["Name"].upper(),
            passenger["Gender"],
            passenger["Seat No"]
        )

        if key not in seen:

            seen.add(key)

            unique.append(passenger)

    return unique


def extract_route(text):

    patterns = [
        r"\bfrom\s+([A-Za-z][A-Za-z .,'()&/-]{1,60})\s+to\s+([A-Za-z][A-Za-z .,'()&/-]{1,60})",
        r"\b([A-Za-z][A-Za-z .,'()&/-]{1,50})\s+[-–—>]\s+([A-Za-z][A-Za-z .,'()&/-]{1,50})\b",
        r"\b([A-Za-z][A-Za-z .,'()&/-]{1,50})\s+TO\s+([A-Za-z][A-Za-z .,'()&/-]{1,50})\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            first = clean_text(
                match.group(1)
            )

            second = clean_text(
                match.group(2)
            )

            if len(first) > 1 and len(second) > 1:
                return first, second

    return "Not detected", "Not detected"


def process_image(image):

    words = ocr_data(image)

    lines = group_lines(words)

    text = all_text_from_lines(lines)

    basic = extract_basic_information(text)

    source, destination = extract_route(text)

    if (
        basic.get("From") == "Not detected"
        and source != "Not detected"
    ):
        basic["From"] = source

    if (
        basic.get("To") == "Not detected"
        and destination != "Not detected"
    ):
        basic["To"] = destination

    passengers = extract_passengers(
        lines,
        image.width
    )

    if not passengers:

        passengers = fallback_passengers(
            lines,
            image.width
        )

    return text, basic, passengers