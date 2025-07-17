#!/bin/bash

echo "🧠 Setup Total-Perspective-Vortex"
echo "=================================="

# Créer environnement virtuel
echo "📦 Création environnement virtuel..."
python3 -m venv venv

# Instructions pour activer
echo ""
echo "✅ Environnement créé !"
echo ""
echo "🔥 ÉTAPES SUIVANTES :"
echo "1. Activer l'environnement :"
echo "   source venv/bin/activate"
echo ""
echo "2. Installer les dépendances :"
echo "   pip install -r requirements-minimal.txt"
echo ""
echo "3. Lancer le projet :"
echo "   cd code"
echo "   python 1-download_data.py"
