"""Formato numérico en español: coma decimal y punto de miles."""


def num(valor: float, decimales: int = 1) -> str:
    """12345.678 -> '12.345,7'."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(proporcion: float, decimales: int = 1) -> str:
    """0.363 -> '36,3 %'."""
    return f"{num(100 * proporcion, decimales)} %"
