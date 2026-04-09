#!/bin/bash

docker restart group-analysis
sleep 2
docker exec -it group-analysis python3 /app/src/main.py restart
#docker exec -it group-analysis bash
