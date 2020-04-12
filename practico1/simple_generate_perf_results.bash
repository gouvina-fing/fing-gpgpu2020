# Comentar las invocaciones de corrida_por_filas y corrida_por_bloques en ej3.c y compilar antes de correr este script

export EVENT_FLAGS="-e task-clock,context-switches,cpu-migrations,page-faults,cycles,stalled-cycles-frontend,stalled-cycles-backend,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-load-misses"

#./ej3.x m n p nb
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/simple_2048_2048_2048.txt ./ej3.x 2048 2048 2048 512
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/simple_2048_2048_1024.txt ./ej3.x 2048 2048 1024 512
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/simple_1024_1024_2048.txt ./ej3.x 1024 1024 2048 512