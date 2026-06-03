#!/bin/bash

SERVER="jorge@192.168.2.4"
REMOTE_DIR="/work/jorge/GroupAnalysisApp"
TMUX_SESSION="ServicioRemote"
TMUX_WINDOW="main"
COMMAND="python3 src/main.py"
CONTAINER="group-analysis"

echo "Haciendo push a Github..."
sleep 1
git push origin main || { echo "Error: push falló, abortando despliegue"; exit 1; }

echo "Conectando al servidor, actualizando código y reiniciando proceso en el contenedor..."
echo ""
sleep 1

ssh $SERVER "
    cd $REMOTE_DIR || { echo 'Error: no se pudo acceder a $REMOTE_DIR'; exit 1; }
    git pull origin main || { echo 'Error: git pull falló'; exit 1; }
    
    bash BaseCode/restart_container.sh || { echo 'Error: no se pudo reiniciar el contenedor'; exit 1; }
    sleep 2

    docker exec $CONTAINER tmux new-session -d -s $TMUX_SESSION -n $TMUX_WINDOW \"$COMMAND; bash\"
"
echo "Despliegue completado."