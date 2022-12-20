
#Evaluation
PATH=results/evaluation
for RUN in 1 2 3 4; do # 1 2 3 4 5 6 7 8
  for ENV in "Train"; do # # "Maze7" "Maze15"  "Mazes7" "Mazes15"
    for ALG in "PPO" "PPO CROP Action" "PPO CROP Object" "PPO CROP Radius"; do # "DQN"  "A2C"
      echo "Running $ALG $RUN"
      nohup python -m run $ALG --env DistributionalShift $ENV --path $PATH &> "$PATH-out/$ENV-$ALG-$RUN.out" &
    done
    # sleep 30m
  done
done

#Benchmark
PATH=results/benchmark
for RUN in 1 2 3 4; do # 1 2 3 4 5 6 7 8
  for ENV in "Maze9"; do # # "Maze7" "Maze15" "Mazes7" "Mazes9" "Mazes15"
    for ALG in "PPO" "PPO CROP Action" "PPO CROP Object" "PPO CROP Radius"; do # TODO SAC, ("DQN"  "A2C")
      echo "Running $ALG $RUN"
      nohup python -m run $ALG --env DistributionalShift $ENV --path &> "$PATH-out/$ENV-$ALG-$RUN.out" &
    done
    # sleep 30m
  done
done

# nohup python -m run PPO CROP Action --env DistributionalShift Maze7 &> "out/Maze7-PPO CROP Action-4.out" &