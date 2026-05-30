#!/bin/bash

SERVER="jorge@192.168.2.4"
REMOTE_DIR="/work/jorge/GroupAnalysisApp"
TMUX_SESSION="ServicioRemote"
TMUX_WINDOW="main"
COMMAND="python3 src/main.py"

echo "Haciendo push a Github..."
git push origin main

echo "Conectando al servidor y ejecutando el comando..."
ssh $SERVER "cd $REMOTE_DIR && git pull && cd BaseCode&& tmux send-keys -t ${TMUX_SESSION}:${TMUX_WINDOW} C-c '${COMMAND}' Enter"

echo "Despliegue completado."