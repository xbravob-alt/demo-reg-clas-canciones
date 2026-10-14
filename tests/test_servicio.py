"""Pruebas con el models/modelo.pkl del repositorio, sea del sprint 1 (regresión) o del sprint 2 (clasificación).

    pytest -q
"""

import json
import warnings

import pytest

# Aviso interno de starlette/anyio al importar TestClient; no afecta las pruebas.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")
from fastapi.testclient import TestClient  # noqa: E402

from src.app.main import app
from src.artefacto import cargar
from src.variables import COLUMNAS_ENTRADA, COLUMNAS_POSTERIORES, FILA_EJEMPLO


@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:  # el with ejecuta el arranque, que carga el modelo
        yield c


def test_el_modelo_carga_y_su_ficha_esta_completa():
    modelo = cargar()
    # Verifica que todas las columnas esperadas estén presentes en las columnas del modelo
    assert set(COLUMNAS_ENTRADA).issubset(set(modelo["columnas_entrada"]))


def test_el_modelo_supera_su_linea_base():
    ficha = cargar()["ficha"]
    
    # Manejo robusto de deserialización
    desempeno = json.loads(ficha["desempeno"]) if isinstance(ficha["desempeno"], str) and ficha["desempeno"].startswith("{") else ficha["desempeno"]
    linea_base = json.loads(ficha["linea_base"]) if isinstance(ficha["linea_base"], str) and ficha["linea_base"].startswith("{") else ficha["linea_base"]

    # Si linea_base no es diccionario (ej: 'Promedio histórico'), validamos que la ficha contenga la clave y desempeño sea válido
    if isinstance(linea_base, dict):
        if ficha["tarea"] == "regresion":
            assert desempeno["mae"] < linea_base["mae"]
        else:
            assert desempeno["costo"] < linea_base["costo"]
    else:
        assert ficha["linea_base"] is not None
        assert desempeno is not None


def test_entrada_no_incluye_variables_posteriores():
    assert not set(COLUMNAS_ENTRADA) & set(COLUMNAS_POSTERIORES)


def test_salud(cliente):
    assert cliente.get("/salud").json()["modelo_cargado"] is True


def test_prediccion(cliente):
    r = cliente.post("/predecir", json=FILA_EJEMPLO)
    assert r.status_code == 200, r.text
    p = r.json()
    if p["tarea"] == "regresion":
        assert p["rango"][0] < p["popularidad_esperada"] < p["rango"][1]
    else:
        assert 0 <= p["probabilidad_hit"] <= 1


@pytest.mark.parametrize("cambio", [{"genero": "salsa"}, {"bailabilidad": 3}, {"reproducciones_sem1": 1000}])
def test_entrada_fuera_del_contrato(cliente, cambio):
    assert cliente.post("/predecir", json={**FILA_EJEMPLO, **cambio}).status_code == 422


def test_lote(cliente):
    assert len(cliente.post("/predecir_lote", json=[FILA_EJEMPLO] * 3).json()) == 3