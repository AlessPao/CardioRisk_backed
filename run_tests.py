#!/usr/bin/env python3
"""
Script para ejecutar pruebas unitarias y generar reportes completos
Diseñado para capturas de pantalla del informe técnico
"""

import subprocess
import sys
import os
from datetime import datetime

def run_tests_with_reports():
    """Ejecutar todas las pruebas con diferentes formatos de reporte"""
    
    print("🧪 INICIANDO SUITE DE PRUEBAS UNITARIAS")
    print("="*60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Framework: pytest + FastAPI TestClient")
    print("="*60)
    
    # 1. Ejecutar pruebas básicas con output detallado
    print("\n📊 1. EJECUTANDO PRUEBAS CON OUTPUT DETALLADO...")
    cmd1 = [
        sys.executable, "-m", "pytest", 
        "tests/test_api.py", 
        "-v", "-s", 
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result1 = subprocess.run(cmd1, capture_output=False, text=True, cwd=".")
        print(f"✅ Pruebas básicas completadas (Exit code: {result1.returncode})")
    except Exception as e:
        print(f"❌ Error en pruebas básicas: {e}")
    
    # 2. Generar reporte de cobertura HTML
    print("\n📈 2. GENERANDO REPORTE DE COBERTURA...")
    cmd2 = [
        sys.executable, "-m", "pytest",
        "tests/test_api.py",
        "--cov=app",
        "--cov-report=html:htmlcov",
        "--cov-report=term",
        "--cov-report=json:coverage.json",
        "-v"
    ]
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=".")
        print(f"✅ Reporte de cobertura generado (Exit code: {result2.returncode})")
        print("   📁 HTML Report: htmlcov/index.html")
        print("   📄 JSON Report: coverage.json")
    except Exception as e:
        print(f"❌ Error en reporte de cobertura: {e}")
    
    # 3. Generar reporte JUnit XML
    print("\n📋 3. GENERANDO REPORTE JUNIT XML...")
    cmd3 = [
        sys.executable, "-m", "pytest",
        "tests/test_api.py",
        "--junitxml=test_report.xml",
        "-v"
    ]
    
    try:
        result3 = subprocess.run(cmd3, capture_output=True, text=True, cwd=".")
        print(f"✅ Reporte JUnit XML generado (Exit code: {result3.returncode})")
        print("   📄 XML Report: test_report.xml")
    except Exception as e:
        print(f"❌ Error en reporte JUnit: {e}")
    
    # 4. Ejecutar pruebas por categorías
    print("\n🎯 4. EJECUTANDO PRUEBAS POR CATEGORÍAS...")
    
    categories = [
        ("Endpoints", "tests/test_api.py::TestEndpoints", "🌐"),
        ("Predicción", "tests/test_api.py::TestPredictionProfiles", "🫀"),
        ("Validación", "tests/test_api.py::TestInputValidation", "✅"),
        ("Estructura", "tests/test_api.py::TestResponseStructure", "📊"),
        ("Rendimiento", "tests/test_api.py::TestPerformance", "⚡")
    ]
    
    for name, pattern, icon in categories:
        print(f"\n{icon} Ejecutando pruebas de {name}...")
        cmd_cat = [
            sys.executable, "-m", "pytest",
            pattern,
            "-v", "-s",
            "--tb=short"
        ]
        
        try:
            result_cat = subprocess.run(cmd_cat, capture_output=False, text=True, cwd=".")
            status = "✅ EXITOSO" if result_cat.returncode == 0 else "❌ FALLIDO"
            print(f"   {status} (Exit code: {result_cat.returncode})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("🎉 SUITE DE PRUEBAS COMPLETADA")
    print("="*60)
    print("📁 Archivos generados:")
    print("   • htmlcov/index.html    (Reporte de cobertura HTML)")
    print("   • coverage.json         (Datos de cobertura JSON)")
    print("   • test_report.xml       (Reporte JUnit XML)")
    print("="*60)

if __name__ == "__main__":
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("app") or not os.path.exists("tests"):
        print("❌ Error: Ejecutar desde el directorio raíz del proyecto")
        sys.exit(1)
      # Verificar que pytest está instalado
    try:
        import pytest
    except ImportError:
        print("❌ Error: Instalar pytest y pytest-cov")
        print("💡 Ejecutar: pip install pytest pytest-cov")
        sys.exit(1)
    
    run_tests_with_reports()
