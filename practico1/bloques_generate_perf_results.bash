# Comentar las invocaciones de corrida_simple y corrida_por_filas en ej3.c y compilar antes de correr este script

export EVENT_FLAGS="-e task-clock,context-switches,cpu-migrations,page-faults,cycles,stalled-cycles-frontend,stalled-cycles-backend,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,L1-icache-load-misses"

#./ej3.x m n p nb

sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_2048_64.txt ./ej3.x 2048 2048 2048 64
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_1024_64.txt ./ej3.x 2048 2048 1024 64
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_1024_1024_2048_64.txt ./ej3.x 1024 1024 2048 64

sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_2048_128.txt ./ej3.x 2048 2048 2048 128
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_1024_128.txt ./ej3.x 2048 2048 1024 128
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_1024_1024_2048_128.txt ./ej3.x 1024 1024 2048 128

sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_2048_256.txt ./ej3.x 2048 2048 2048 256
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_1024_256.txt ./ej3.x 2048 2048 1024 256
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_1024_1024_2048_256.txt ./ej3.x 1024 1024 2048 256

sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_2048_512.txt ./ej3.x 2048 2048 2048 512
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_2048_2048_1024_512.txt ./ej3.x 2048 2048 1024 512
sudo perf stat $EVENT_FLAGS -o resultados/renzo/perf/bloques_1024_1024_2048_512.txt ./ej3.x 1024 1024 2048 512