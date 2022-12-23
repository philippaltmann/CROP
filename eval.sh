for RUN in 8; do # 1 2 3 4 5 6 7 8
  for ALG in "PPO" "PPO CROP Action" "PPO CROP Object" "PPO CROP Radius"; do
    echo "Running $ALG $RUN"
    nohup python -m run $ALG --env Train --path results/1-evaluation &> "results/out/$ENV-$ALG-$RUN.out" &
  done
done