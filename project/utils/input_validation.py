from typing import Tuple


def detect_language(text: str) -> str:
    """
    returns 'rus' if most letters are Cyrillic, else 'other'.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "other"

    cyr = 0
    for ch in letters:
        low = ch.lower()
        if ("а" <= low <= "я") or (low == "ё"):
            cyr += 1

    return "rus" if (cyr / len(letters)) >= 0.6 else "other"


def check_correct_input(user_input: str) -> Tuple[bool, str, str, str, str]:
    """
    Parse and validate input.
    Expected format: "<query> <index_type> <impl>"
    Example: "борщ F dict"
    Returns: ok, query, index_type, impl, error_message
    """
    if not user_input or not user_input.strip():
        return False, "", "", "", "Пустой ввод. Пример: борщ F dict"

    parts = user_input.split()
    if len(parts) != 3:
        return False, "", "", "", "Формат: <запрос> <F|BM> <dict|matrix>. Пример: борщ F dict"

    query = parts[0]
    index_type = parts[1].upper()
    impl = parts[2].lower()

    if detect_language(query) != "rus":
        return False, "", "", "", "Кажется, запрос не на русском языке"

    if index_type not in ("F", "BM"):
        return False, "", "", "", "Тип индекса: 'F' или 'BM' (английскими буквами)"

    if impl not in ("dict", "matrix"):
        return False, "", "", "", "Реализация: 'dict' или 'matrix'"

    return True, query, index_type, impl, ""


def parse_user_input(user_input: str) -> Tuple[bool, str, str, str, str]:
    """
    Parse and validate input.

    Expected format: "<query...> <F|BM> <dict|matrix>"
    Example: "борщ чеснок BM matrix"
    """
    if not user_input or not user_input.strip():
        return False, "", "", "", "Пустой ввод. Пример: полезные сладости BM matrix"

    parts = user_input.split()
    if len(parts) < 3:
        return (
            False,
            "",
            "",
            "",
            "Формат: <запрос...> <F|BM> <dict|matrix>. Пример: полезные сладости BM matrix",
        )

    query = " ".join(parts[:-2]).strip()
    index_type = parts[-2].upper()
    impl = parts[-1].lower()

    if not query:
        return False, "", "", "", "Пустой запрос. Пример: полезные сладости BM matrix"

    if detect_language(query) != "rus":
        return False, "", "", "", "Кажется, запрос не на русском языке"

    if index_type not in ("F", "BM"):
        return False, "", "", "", "Тип индекса: 'F' или 'BM' (английскими буквами)"

    if impl not in ("dict", "matrix"):
        return False, "", "", "", "Реализация: 'dict' или 'matrix'"

    return True, query, index_type, impl, ""


def read_user_input() -> tuple[str, str, str]:
    """Ask for input until it is valid."""
    while True:
        user_input = input(
            "Введите: <запрос...> <F|BM> <dict|matrix> (пример: полезные сладости BM matrix): "
        ).strip()

        ok, query, index_type, impl, msg = parse_user_input(user_input)
        if ok:
            return query, index_type, impl

        print("\n" + msg + "\n")