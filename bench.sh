# #Benchmark
for RUN in 2 3 4 5 6 7 8; do
  for METHOD in "PPO" "PPO CROP Action" "PPO CROP Object" "PPO CROP Radius" "PPO RAD" "A2C"; do
    for ENV in Maze Mazes; do
      for SIZE in 7 11; do
        O="results/out/$ENV-$SIZE/$METHOD"; mkdir -p "$O"
        nohup echo "Running $METHOD $ENV $SIZE $RUN" &> "$O/$RUN.out" &
        nohup python -m run $METHOD --env $ENV$SIZE --path results/2-benchmark &> "$O/$RUN.out" &
        sleep 1s
      done
    done
    sleep 1h
  done
done
