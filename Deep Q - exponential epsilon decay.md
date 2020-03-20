
# Neural Net


```python
def intitializeAgent():
    
    model = Sequential()
    
    model.add(Dense(256,activation='relu',input_shape=(INPUT_SIZE,)))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(OUTPUT_SIZE,activation='linear'))
    
    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    
    return model
```


```python
def getPrediction(env,agent):
    env = env.reshape(INPUT_SIZE,1).reshape(1,-1)
    return agent.predict(env)[0]
```


```python
def getAction(output,invalidActions):
    rand = np.random.rand()
    output[invalidActions] = np.min(output) - 0.1
    
    if rand < EPSILON_RATE ** iteration:
        action = np.random.randint(4)
        while action in invalidActions:
            action = np.random.randint(4)
        return action
    else:
        return np.argmax(output)
```


```python
def getGreedyAction(output,invalidActions):
    output[invalidActions] = np.min(output) - 0.1
    return np.argmax(output)
```


```python
def train(initialStateMemory,actionMemory,rewardMemory,finalStateMemory,model,epochs):
    
    if memCount < MAX_MEM_SIZE:
        sample = np.random.choice(memCount, SAMPLE_SIZE)
    else:
        sample = np.random.choice(MAX_MEM_SIZE, SAMPLE_SIZE)
    batchSize = sample.shape[0] // BATCH_SIZE
    
    if batchSize == 0:
        batchSize = 1
                
    targetQs = getTargetQs(initialStateMemory[sample],
                            actionMemory[sample],
                            rewardMemory[sample],
                            finalStateMemory[sample],
                            model)
        
    print("Training Model...")
    model.fit(initialStateMemory[sample],targetQs,batch_size = batchSize, epochs = epochs, verbose = 0)
```

# Replay Memory


```python
def addMemory(initialState, action, reward, finalState, memCount, MAX_MEM_SIZE):
    if memCount < MAX_MEM_SIZE:
        initialStateMemory[memCount] = initialState.reshape(INPUT_SIZE)
        actionMemory[memCount] = action
        rewardMemory[memCount] = reward
        finalStateMemory[memCount] = finalState.reshape(INPUT_SIZE)
    else:
        rand = np.random.randint(MAX_MEM_SIZE)
        initialStateMemory[rand] = initialState.reshape(INPUT_SIZE)
        actionMemory[rand] = action
        rewardMemory[rand] = reward
        finalStateMemory[rand] = finalState.reshape(INPUT_SIZE)
        
    memCount += 1
    
    return initialStateMemory,actionMemory,rewardMemory,finalStateMemory,memCount
```


```python
def getTargetQs(states0, actions0, rewards1, states1, agent):
    DISCOUNT_FACTOR = 0.99
    targetQs = np.zeros((len(states0),OUTPUT_SIZE))
    
    for i in range(len(states0)):
        targetQs[i] = getPrediction(states0[i],agent)
        targetQs[i][int(actions0[i])] = rewards1[i] + DISCOUNT_FACTOR * np.max(getPrediction(states1[i],agent))
    return targetQs
```

# Game Logic


```python
def initializeEnv():
    env = np.zeros((BOARD_LENGTH,BOARD_LENGTH))

    for i in range(2):
        addValue(env)
            
    return env
```


```python
def addValue(env):
    rand = np.random.rand()
    if rand > 0.1:
        value = 2
    else:
        value = 4
        
    coordinate1 = random.sample(range(0,BOARD_LENGTH),1)
    coordinate2 = random.sample(range(0,BOARD_LENGTH),1)
        
    if env[coordinate1,coordinate2] != 0:
        getNewCoordinate(env,value)
    else:
        env[coordinate1,coordinate2] = value
        
    return env
```


```python
def getNewCoordinate(env,value):
    
    coordinate1 = random.sample(range(0,BOARD_LENGTH),1)
    coordinate2 = random.sample(range(0,BOARD_LENGTH),1)
    
    while env[coordinate1,coordinate2] != 0:
        coordinate1 = random.sample(range(0,BOARD_LENGTH),1)
        coordinate2 = random.sample(range(0,BOARD_LENGTH),1)
    
    env[coordinate1,coordinate2] = value
```


```python
def step(env,action,score):
    
    if action == 0:
        env, score, validAction = actionUp(env,score)
    if action == 1:
        env, score, validAction = actionDown(env,score)
    if action == 2:
        env, score, validAction = actionLeft(env,score)
    if action == 3:
        env, score, validAction = actionRight(env,score)
    
    return env, score, validAction
```


```python
def actionUp(env,score):
    validAction = False
    
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            row = i
            while env[row][j] != 0 and row != 0:
                if env[row - 1][j] != 0:
                    if env[row - 1][j] == env[row][j]:
                        env[row - 1][j] *= 2
                        env[row][j] = 0
                        score += env[row - 1][j]
                        validAction = True
                    break
                temp = env[row][j]
                env[row-1][j] = temp
                env[row][j] = 0
                row -= 1
                validAction = True
    return env, score, validAction
```


```python
def actionDown(env,score):
    validAction = False
    
    for i in range(3,-1,-1):
        for j in range(BOARD_LENGTH):
            row = i
            while env[row][j] != 0 and row != 3:
                if env[row + 1][j] != 0:
                    if env[row + 1][j] == env[row][j]:
                        env[row + 1][j] *= 2
                        env[row][j] = 0
                        score += env[row + 1][j]
                        validAction = True
                    break
                temp = env[row][j]
                env[row + 1][j] = temp
                env[row][j] = 0
                row += 1
                validAction = True
    return env, score, validAction
```


```python
def actionLeft(env,score):
    validAction = False
    env = env.T
    
    for i in range(BOARD_LENGTH):
        for j in range(BOARD_LENGTH):
            row = i
            while env[row][j] != 0 and row != 0:
                if env[row - 1][j] != 0:
                    if env[row - 1][j] == env[row][j]:
                        env[row - 1][j] *= 2
                        env[row][j] = 0
                        score += env[row - 1][j]
                        validAction = True
                    break
                temp = env[row][j]
                env[row-1][j] = temp
                env[row][j] = 0
                row -= 1
                validAction = True
    env = env.T
    return env, score, validAction
```


```python
def actionRight(env,score):
    validAction = False
    env = env.T
    
    for i in range(3,-1,-1):
        for j in range(BOARD_LENGTH):
            row = i
            while env[row][j] != 0 and row != 3:
                if env[row + 1][j] != 0:
                    if env[row + 1][j] == env[row][j]:
                        env[row + 1][j] *= 2
                        env[row][j] = 0
                        score += env[row + 1][j]
                        validAction = True
                    break
                temp = env[row][j]
                env[row + 1][j] = temp
                env[row][j] = 0
                row += 1
                validAction = True
    env = env.T
    return env, score, validAction
```


```python
def getAvailableAction(env):
    tempEnv = np.copy(env)
    available = False
    
    for i in range(tempEnv.shape[0]):
        for j in range(tempEnv.shape[0] - 1):
            if tempEnv[i][j] == tempEnv[i][j+1]:
                available = True
                
                
    tempEnv = tempEnv.T
    for i in range(tempEnv.shape[0]):
        for j in range(tempEnv.shape[0] - 1):
            if tempEnv[i][j] == tempEnv[i][j+1]:
                available = True
                break
    
    return available
```


```python
def run(env,agent,memCount,testFlag,score = 0):
    
    gameOver = False
    while gameOver == False:
        invalidActions = []
        validAction = False
        while validAction == False:
            
            if not testFlag:
                previousEnv = np.copy(env)
                previousScore = np.copy(score)
            
            output = getPrediction(env,agent)
            if testFlag: 
                action = getGreedyAction(output,invalidActions) 
            else: 
                action = getAction(output,invalidActions)
            invalidActions.append(action)
            env, score, validAction = step(env, action, score)
        
        if not testFlag:
            initialStateMemory,actionMemory,rewardMemory,finalStateMemory,memCount = addMemory(previousEnv, action, score - previousScore, env, memCount, MAX_MEM_SIZE)

        env = addValue(env)
        
        if np.sum(env == 0) == 0:
            gameOver = not getAvailableAction(env)
    
    if not testFlag:
        return score,initialStateMemory,actionMemory,rewardMemory,finalStateMemory,memCount
    else: 
        return score
```

# Main Output Functions


```python
def printIter():
    print("\n === === === Iteration",iteration,"of",ITERATIONS,'=== === ===\n')
```


```python
def printStats(timeLog,iteration,ITERATIONS,memCount,scoreLog,EPISODES):
    print("\nEstimated time remaining: ", format(np.mean(timeLog) * (ITERATIONS - iteration) / 60,'.2f')," Minutes\n")
    print("Memories Elapsed: ",memCount)
    print("Current Epsilon: ",format(EPSILON_RATE ** iteration,'.2f'))
    print("Ave Score for Epoch: ", format(np.mean(scoreLog[len(scoreLog) - EPISODES : len(scoreLog)-1]) ,'.2f'))
```


```python
def printTrainInfo():
    print("Maximum memory size:",MAX_MEM_SIZE)
    print("Sample size for training:",SAMPLE_SIZE)
    print("Batch size from sample:",BATCH_SIZE)
    print("Epslon Rate:",EPSILON_RATE)
```

# Main


```python
import numpy as np 
import matplotlib.pyplot as plt
import random
import time
import sys

from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Conv2D

from sklearn.linear_model import LinearRegression
```


```python
BOARD_LENGTH = 4
INPUT_SIZE = 16
OUTPUT_SIZE = BOARD_LENGTH

memCount = 0
MAX_MEM_SIZE = 200_000
SAMPLE_SIZE = MAX_MEM_SIZE // 200
BATCH_SIZE = SAMPLE_SIZE // 100
initialStateMemory = np.zeros((MAX_MEM_SIZE,INPUT_SIZE))
actionMemory = np.zeros(MAX_MEM_SIZE)
rewardMemory = np.zeros(MAX_MEM_SIZE)
finalStateMemory = np.zeros((MAX_MEM_SIZE,INPUT_SIZE))


gameScoreLog = []
greedyScoreLog = []
agent = intitializeAgent()
```


```python
ITERATIONS = 200
EPISODES = 50
TESTS = EPISODES // 5
EPOCHS = 4
MIN_EPSILON = 0.01
EPSILON_RATE = MIN_EPSILON ** (1./ITERATIONS)
```


```python
startTime = time.time()
timeLog = []

printTrainInfo()

for iteration in range(ITERATIONS):
    printIter()
    iterationStart = time.time()
    
    for i in range(EPISODES):

        score,initialStateMemory,actionMemory,rewardMemory,finalStateMemory,memCount = run(initializeEnv(),agent,memCount,False)
        gameScoreLog.append(score)
    
    for i in range(TESTS):
        
        score = run(initializeEnv(),agent,memCount,True)
        greedyScoreLog.append(score)
        
    train(initialStateMemory,actionMemory,rewardMemory,finalStateMemory,agent,EPOCHS)
    timeLog.append(time.time() - iterationStart)
    
    printStats(timeLog,iteration,ITERATIONS,memCount,gameScoreLog,EPISODES)
```

    Maximum memory size: 200000
    Sample size for training: 1000
    Batch size from sample: 10
    Epslon Rate: 0.9772372209558107
    
     === === === Iteration 0 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  27.98  Minutes
    
    Memories Elapsed:  6543
    Current Epsilon:  1.00
    Ave Score for Epoch:  1280.57
    
     === === === Iteration 1 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.50  Minutes
    
    Memories Elapsed:  12385
    Current Epsilon:  0.98
    Ave Score for Epoch:  1085.22
    
     === === === Iteration 2 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.49  Minutes
    
    Memories Elapsed:  18428
    Current Epsilon:  0.95
    Ave Score for Epoch:  1135.67
    
     === === === Iteration 3 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.03  Minutes
    
    Memories Elapsed:  24773
    Current Epsilon:  0.93
    Ave Score for Epoch:  1250.04
    
     === === === Iteration 4 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.19  Minutes
    
    Memories Elapsed:  30585
    Current Epsilon:  0.91
    Ave Score for Epoch:  1076.33
    
     === === === Iteration 5 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.96  Minutes
    
    Memories Elapsed:  36663
    Current Epsilon:  0.89
    Ave Score for Epoch:  1152.57
    
     === === === Iteration 6 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.78  Minutes
    
    Memories Elapsed:  42558
    Current Epsilon:  0.87
    Ave Score for Epoch:  1069.80
    
     === === === Iteration 7 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.79  Minutes
    
    Memories Elapsed:  49031
    Current Epsilon:  0.85
    Ave Score for Epoch:  1214.94
    
     === === === Iteration 8 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.71  Minutes
    
    Memories Elapsed:  55509
    Current Epsilon:  0.83
    Ave Score for Epoch:  1230.69
    
     === === === Iteration 9 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.65  Minutes
    
    Memories Elapsed:  61517
    Current Epsilon:  0.81
    Ave Score for Epoch:  1154.78
    
     === === === Iteration 10 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.54  Minutes
    
    Memories Elapsed:  67766
    Current Epsilon:  0.79
    Ave Score for Epoch:  1204.00
    
     === === === Iteration 11 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.32  Minutes
    
    Memories Elapsed:  73690
    Current Epsilon:  0.78
    Ave Score for Epoch:  1084.57
    
     === === === Iteration 12 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.30  Minutes
    
    Memories Elapsed:  79819
    Current Epsilon:  0.76
    Ave Score for Epoch:  1156.33
    
     === === === Iteration 13 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.36  Minutes
    
    Memories Elapsed:  85917
    Current Epsilon:  0.74
    Ave Score for Epoch:  1152.49
    
     === === === Iteration 14 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.34  Minutes
    
    Memories Elapsed:  92283
    Current Epsilon:  0.72
    Ave Score for Epoch:  1248.98
    
     === === === Iteration 15 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.21  Minutes
    
    Memories Elapsed:  97967
    Current Epsilon:  0.71
    Ave Score for Epoch:  1041.96
    
     === === === Iteration 16 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.01  Minutes
    
    Memories Elapsed:  104036
    Current Epsilon:  0.69
    Ave Score for Epoch:  1139.02
    
     === === === Iteration 17 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.78  Minutes
    
    Memories Elapsed:  109613
    Current Epsilon:  0.68
    Ave Score for Epoch:  1011.67
    
     === === === Iteration 18 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.70  Minutes
    
    Memories Elapsed:  115919
    Current Epsilon:  0.66
    Ave Score for Epoch:  1228.65
    
     === === === Iteration 19 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.60  Minutes
    
    Memories Elapsed:  122248
    Current Epsilon:  0.65
    Ave Score for Epoch:  1198.37
    
     === === === Iteration 20 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.42  Minutes
    
    Memories Elapsed:  128063
    Current Epsilon:  0.63
    Ave Score for Epoch:  1071.76
    
     === === === Iteration 21 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.24  Minutes
    
    Memories Elapsed:  133642
    Current Epsilon:  0.62
    Ave Score for Epoch:  1007.18
    
     === === === Iteration 22 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.13  Minutes
    
    Memories Elapsed:  139657
    Current Epsilon:  0.60
    Ave Score for Epoch:  1113.06
    
     === === === Iteration 23 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.99  Minutes
    
    Memories Elapsed:  145258
    Current Epsilon:  0.59
    Ave Score for Epoch:  996.82
    
     === === === Iteration 24 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.88  Minutes
    
    Memories Elapsed:  151410
    Current Epsilon:  0.58
    Ave Score for Epoch:  1120.98
    
     === === === Iteration 25 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.80  Minutes
    
    Memories Elapsed:  157549
    Current Epsilon:  0.56
    Ave Score for Epoch:  1155.51
    
     === === === Iteration 26 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.73  Minutes
    
    Memories Elapsed:  163672
    Current Epsilon:  0.55
    Ave Score for Epoch:  1149.96
    
     === === === Iteration 27 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.71  Minutes
    
    Memories Elapsed:  169854
    Current Epsilon:  0.54
    Ave Score for Epoch:  1168.65
    
     === === === Iteration 28 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.67  Minutes
    
    Memories Elapsed:  176016
    Current Epsilon:  0.52
    Ave Score for Epoch:  1143.02
    
     === === === Iteration 29 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.54  Minutes
    
    Memories Elapsed:  182139
    Current Epsilon:  0.51
    Ave Score for Epoch:  1156.82
    
     === === === Iteration 30 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.35  Minutes
    
    Memories Elapsed:  187660
    Current Epsilon:  0.50
    Ave Score for Epoch:  982.04
    
     === === === Iteration 31 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.28  Minutes
    
    Memories Elapsed:  193411
    Current Epsilon:  0.49
    Ave Score for Epoch:  1022.53
    
     === === === Iteration 32 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.16  Minutes
    
    Memories Elapsed:  199177
    Current Epsilon:  0.48
    Ave Score for Epoch:  1032.16
    
     === === === Iteration 33 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.09  Minutes
    
    Memories Elapsed:  205697
    Current Epsilon:  0.47
    Ave Score for Epoch:  1227.10
    
     === === === Iteration 34 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.04  Minutes
    
    Memories Elapsed:  211442
    Current Epsilon:  0.46
    Ave Score for Epoch:  1048.82
    
     === === === Iteration 35 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.95  Minutes
    
    Memories Elapsed:  217711
    Current Epsilon:  0.45
    Ave Score for Epoch:  1176.82
    
     === === === Iteration 36 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.75  Minutes
    
    Memories Elapsed:  223282
    Current Epsilon:  0.44
    Ave Score for Epoch:  991.67
    
     === === === Iteration 37 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.58  Minutes
    
    Memories Elapsed:  229114
    Current Epsilon:  0.43
    Ave Score for Epoch:  1077.39
    
     === === === Iteration 38 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.45  Minutes
    
    Memories Elapsed:  235360
    Current Epsilon:  0.42
    Ave Score for Epoch:  1192.33
    
     === === === Iteration 39 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.30  Minutes
    
    Memories Elapsed:  241330
    Current Epsilon:  0.41
    Ave Score for Epoch:  1076.90
    
     === === === Iteration 40 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.17  Minutes
    
    Memories Elapsed:  247682
    Current Epsilon:  0.40
    Ave Score for Epoch:  1222.12
    
     === === === Iteration 41 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.05  Minutes
    
    Memories Elapsed:  253888
    Current Epsilon:  0.39
    Ave Score for Epoch:  1184.73
    
     === === === Iteration 42 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.00  Minutes
    
    Memories Elapsed:  260471
    Current Epsilon:  0.38
    Ave Score for Epoch:  1285.47
    
     === === === Iteration 43 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.97  Minutes
    
    Memories Elapsed:  266945
    Current Epsilon:  0.37
    Ave Score for Epoch:  1231.59
    
     === === === Iteration 44 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.87  Minutes
    
    Memories Elapsed:  272647
    Current Epsilon:  0.36
    Ave Score for Epoch:  1038.37
    
     === === === Iteration 45 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.78  Minutes
    
    Memories Elapsed:  278844
    Current Epsilon:  0.35
    Ave Score for Epoch:  1155.35
    
     === === === Iteration 46 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.72  Minutes
    
    Memories Elapsed:  284922
    Current Epsilon:  0.35
    Ave Score for Epoch:  1110.61
    
     === === === Iteration 47 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.66  Minutes
    
    Memories Elapsed:  290691
    Current Epsilon:  0.34
    Ave Score for Epoch:  1033.31
    
     === === === Iteration 48 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.57  Minutes
    
    Memories Elapsed:  296986
    Current Epsilon:  0.33
    Ave Score for Epoch:  1216.08
    
     === === === Iteration 49 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.52  Minutes
    
    Memories Elapsed:  303509
    Current Epsilon:  0.32
    Ave Score for Epoch:  1257.71
    
     === === === Iteration 50 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.42  Minutes
    
    Memories Elapsed:  309707
    Current Epsilon:  0.32
    Ave Score for Epoch:  1164.41
    
     === === === Iteration 51 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.33  Minutes
    
    Memories Elapsed:  315742
    Current Epsilon:  0.31
    Ave Score for Epoch:  1103.76
    
     === === === Iteration 52 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.20  Minutes
    
    Memories Elapsed:  321363
    Current Epsilon:  0.30
    Ave Score for Epoch:  1005.88
    
     === === === Iteration 53 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.13  Minutes
    
    Memories Elapsed:  327040
    Current Epsilon:  0.30
    Ave Score for Epoch:  994.20
    
     === === === Iteration 54 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.03  Minutes
    
    Memories Elapsed:  332864
    Current Epsilon:  0.29
    Ave Score for Epoch:  1076.98
    
     === === === Iteration 55 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.95  Minutes
    
    Memories Elapsed:  339266
    Current Epsilon:  0.28
    Ave Score for Epoch:  1220.73
    
     === === === Iteration 56 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.88  Minutes
    
    Memories Elapsed:  345179
    Current Epsilon:  0.28
    Ave Score for Epoch:  1062.69
    
     === === === Iteration 57 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.78  Minutes
    
    Memories Elapsed:  351428
    Current Epsilon:  0.27
    Ave Score for Epoch:  1168.16
    
     === === === Iteration 58 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.68  Minutes
    
    Memories Elapsed:  357755
    Current Epsilon:  0.26
    Ave Score for Epoch:  1179.43
    
     === === === Iteration 59 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.56  Minutes
    
    Memories Elapsed:  363831
    Current Epsilon:  0.26
    Ave Score for Epoch:  1145.06
    
     === === === Iteration 60 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.47  Minutes
    
    Memories Elapsed:  370185
    Current Epsilon:  0.25
    Ave Score for Epoch:  1207.59
    
     === === === Iteration 61 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.40  Minutes
    
    Memories Elapsed:  376718
    Current Epsilon:  0.25
    Ave Score for Epoch:  1224.82
    
     === === === Iteration 62 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.36  Minutes
    
    Memories Elapsed:  383513
    Current Epsilon:  0.24
    Ave Score for Epoch:  1296.90
    
     === === === Iteration 63 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.31  Minutes
    
    Memories Elapsed:  389801
    Current Epsilon:  0.23
    Ave Score for Epoch:  1179.02
    
     === === === Iteration 64 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.21  Minutes
    
    Memories Elapsed:  396193
    Current Epsilon:  0.23
    Ave Score for Epoch:  1209.14
    
     === === === Iteration 65 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.12  Minutes
    
    Memories Elapsed:  402568
    Current Epsilon:  0.22
    Ave Score for Epoch:  1187.67
    
     === === === Iteration 66 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.03  Minutes
    
    Memories Elapsed:  408924
    Current Epsilon:  0.22
    Ave Score for Epoch:  1184.98
    
     === === === Iteration 67 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.02  Minutes
    
    Memories Elapsed:  415574
    Current Epsilon:  0.21
    Ave Score for Epoch:  1283.92
    
     === === === Iteration 68 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.93  Minutes
    
    Memories Elapsed:  421757
    Current Epsilon:  0.21
    Ave Score for Epoch:  1147.18
    
     === === === Iteration 69 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.82  Minutes
    
    Memories Elapsed:  428336
    Current Epsilon:  0.20
    Ave Score for Epoch:  1244.16
    
     === === === Iteration 70 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.72  Minutes
    
    Memories Elapsed:  435106
    Current Epsilon:  0.20
    Ave Score for Epoch:  1303.67
    
     === === === Iteration 71 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.61  Minutes
    
    Memories Elapsed:  442042
    Current Epsilon:  0.19
    Ave Score for Epoch:  1319.02
    
     === === === Iteration 72 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.51  Minutes
    
    Memories Elapsed:  448931
    Current Epsilon:  0.19
    Ave Score for Epoch:  1356.08
    
     === === === Iteration 73 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.39  Minutes
    
    Memories Elapsed:  455233
    Current Epsilon:  0.19
    Ave Score for Epoch:  1160.98
    
     === === === Iteration 74 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.29  Minutes
    
    Memories Elapsed:  462364
    Current Epsilon:  0.18
    Ave Score for Epoch:  1404.41
    
     === === === Iteration 75 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.18  Minutes
    
    Memories Elapsed:  469095
    Current Epsilon:  0.18
    Ave Score for Epoch:  1288.90
    
     === === === Iteration 76 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.07  Minutes
    
    Memories Elapsed:  475532
    Current Epsilon:  0.17
    Ave Score for Epoch:  1187.92
    
     === === === Iteration 77 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.97  Minutes
    
    Memories Elapsed:  482511
    Current Epsilon:  0.17
    Ave Score for Epoch:  1380.08
    
     === === === Iteration 78 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.85  Minutes
    
    Memories Elapsed:  488893
    Current Epsilon:  0.17
    Ave Score for Epoch:  1194.78
    
     === === === Iteration 79 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.75  Minutes
    
    Memories Elapsed:  496534
    Current Epsilon:  0.16
    Ave Score for Epoch:  1545.31
    
     === === === Iteration 80 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.64  Minutes
    
    Memories Elapsed:  503220
    Current Epsilon:  0.16
    Ave Score for Epoch:  1264.90
    
     === === === Iteration 81 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.53  Minutes
    
    Memories Elapsed:  509682
    Current Epsilon:  0.15
    Ave Score for Epoch:  1222.20
    
     === === === Iteration 82 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.42  Minutes
    
    Memories Elapsed:  516877
    Current Epsilon:  0.15
    Ave Score for Epoch:  1374.20
    
     === === === Iteration 83 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.32  Minutes
    
    Memories Elapsed:  524485
    Current Epsilon:  0.15
    Ave Score for Epoch:  1536.16
    
     === === === Iteration 84 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.21  Minutes
    
    Memories Elapsed:  531072
    Current Epsilon:  0.14
    Ave Score for Epoch:  1240.41
    
     === === === Iteration 85 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.10  Minutes
    
    Memories Elapsed:  538126
    Current Epsilon:  0.14
    Ave Score for Epoch:  1400.08
    
     === === === Iteration 86 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.02  Minutes
    
    Memories Elapsed:  545838
    Current Epsilon:  0.14
    Ave Score for Epoch:  1533.88
    
     === === === Iteration 87 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.93  Minutes
    
    Memories Elapsed:  553047
    Current Epsilon:  0.13
    Ave Score for Epoch:  1403.92
    
     === === === Iteration 88 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.84  Minutes
    
    Memories Elapsed:  560100
    Current Epsilon:  0.13
    Ave Score for Epoch:  1364.33
    
     === === === Iteration 89 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.75  Minutes
    
    Memories Elapsed:  567828
    Current Epsilon:  0.13
    Ave Score for Epoch:  1543.84
    
     === === === Iteration 90 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.65  Minutes
    
    Memories Elapsed:  574742
    Current Epsilon:  0.13
    Ave Score for Epoch:  1326.53
    
     === === === Iteration 91 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.57  Minutes
    
    Memories Elapsed:  582266
    Current Epsilon:  0.12
    Ave Score for Epoch:  1476.00
    
     === === === Iteration 92 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.53  Minutes
    
    Memories Elapsed:  589374
    Current Epsilon:  0.12
    Ave Score for Epoch:  1386.86
    
     === === === Iteration 93 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.44  Minutes
    
    Memories Elapsed:  596781
    Current Epsilon:  0.12
    Ave Score for Epoch:  1444.57
    
     === === === Iteration 94 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.35  Minutes
    
    Memories Elapsed:  603589
    Current Epsilon:  0.11
    Ave Score for Epoch:  1291.27
    
     === === === Iteration 95 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.28  Minutes
    
    Memories Elapsed:  611040
    Current Epsilon:  0.11
    Ave Score for Epoch:  1488.90
    
     === === === Iteration 96 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.19  Minutes
    
    Memories Elapsed:  618494
    Current Epsilon:  0.11
    Ave Score for Epoch:  1491.84
    
     === === === Iteration 97 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.12  Minutes
    
    Memories Elapsed:  626528
    Current Epsilon:  0.11
    Ave Score for Epoch:  1659.67
    
     === === === Iteration 98 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.05  Minutes
    
    Memories Elapsed:  633810
    Current Epsilon:  0.10
    Ave Score for Epoch:  1439.18
    
     === === === Iteration 99 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.97  Minutes
    
    Memories Elapsed:  641302
    Current Epsilon:  0.10
    Ave Score for Epoch:  1523.92
    
     === === === Iteration 100 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.86  Minutes
    
    Memories Elapsed:  648342
    Current Epsilon:  0.10
    Ave Score for Epoch:  1352.73
    
     === === === Iteration 101 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.84  Minutes
    
    Memories Elapsed:  655881
    Current Epsilon:  0.10
    Ave Score for Epoch:  1500.90
    
     === === === Iteration 102 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.74  Minutes
    
    Memories Elapsed:  663299
    Current Epsilon:  0.10
    Ave Score for Epoch:  1487.92
    
     === === === Iteration 103 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.64  Minutes
    
    Memories Elapsed:  671083
    Current Epsilon:  0.09
    Ave Score for Epoch:  1586.12
    
     === === === Iteration 104 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.54  Minutes
    
    Memories Elapsed:  678388
    Current Epsilon:  0.09
    Ave Score for Epoch:  1462.04
    
     === === === Iteration 105 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.44  Minutes
    
    Memories Elapsed:  685591
    Current Epsilon:  0.09
    Ave Score for Epoch:  1404.98
    
     === === === Iteration 106 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.33  Minutes
    
    Memories Elapsed:  692237
    Current Epsilon:  0.09
    Ave Score for Epoch:  1236.90
    
     === === === Iteration 107 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.22  Minutes
    
    Memories Elapsed:  699801
    Current Epsilon:  0.09
    Ave Score for Epoch:  1510.69
    
     === === === Iteration 108 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.11  Minutes
    
    Memories Elapsed:  707307
    Current Epsilon:  0.08
    Ave Score for Epoch:  1493.22
    
     === === === Iteration 109 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.99  Minutes
    
    Memories Elapsed:  714855
    Current Epsilon:  0.08
    Ave Score for Epoch:  1488.49
    
     === === === Iteration 110 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.88  Minutes
    
    Memories Elapsed:  721958
    Current Epsilon:  0.08
    Ave Score for Epoch:  1344.08
    
     === === === Iteration 111 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.77  Minutes
    
    Memories Elapsed:  729322
    Current Epsilon:  0.08
    Ave Score for Epoch:  1438.20
    
     === === === Iteration 112 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.67  Minutes
    
    Memories Elapsed:  736296
    Current Epsilon:  0.08
    Ave Score for Epoch:  1323.51
    
     === === === Iteration 113 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.57  Minutes
    
    Memories Elapsed:  743828
    Current Epsilon:  0.07
    Ave Score for Epoch:  1494.20
    
     === === === Iteration 114 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.47  Minutes
    
    Memories Elapsed:  751502
    Current Epsilon:  0.07
    Ave Score for Epoch:  1580.82
    
     === === === Iteration 115 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.36  Minutes
    
    Memories Elapsed:  758979
    Current Epsilon:  0.07
    Ave Score for Epoch:  1510.69
    
     === === === Iteration 116 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.26  Minutes
    
    Memories Elapsed:  766835
    Current Epsilon:  0.07
    Ave Score for Epoch:  1596.24
    
     === === === Iteration 117 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.16  Minutes
    
    Memories Elapsed:  774364
    Current Epsilon:  0.07
    Ave Score for Epoch:  1466.94
    
     === === === Iteration 118 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.05  Minutes
    
    Memories Elapsed:  782028
    Current Epsilon:  0.07
    Ave Score for Epoch:  1526.45
    
     === === === Iteration 119 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.94  Minutes
    
    Memories Elapsed:  789064
    Current Epsilon:  0.06
    Ave Score for Epoch:  1349.55
    
     === === === Iteration 120 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.83  Minutes
    
    Memories Elapsed:  796064
    Current Epsilon:  0.06
    Ave Score for Epoch:  1382.78
    
     === === === Iteration 121 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.71  Minutes
    
    Memories Elapsed:  804078
    Current Epsilon:  0.06
    Ave Score for Epoch:  1625.71
    
     === === === Iteration 122 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.61  Minutes
    
    Memories Elapsed:  812279
    Current Epsilon:  0.06
    Ave Score for Epoch:  1641.39
    
     === === === Iteration 123 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.50  Minutes
    
    Memories Elapsed:  819712
    Current Epsilon:  0.06
    Ave Score for Epoch:  1471.76
    
     === === === Iteration 124 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.39  Minutes
    
    Memories Elapsed:  827039
    Current Epsilon:  0.06
    Ave Score for Epoch:  1459.76
    
     === === === Iteration 125 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.33  Minutes
    
    Memories Elapsed:  835651
    Current Epsilon:  0.06
    Ave Score for Epoch:  1836.65
    
     === === === Iteration 126 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.21  Minutes
    
    Memories Elapsed:  843123
    Current Epsilon:  0.05
    Ave Score for Epoch:  1500.65
    
     === === === Iteration 127 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.09  Minutes
    
    Memories Elapsed:  850156
    Current Epsilon:  0.05
    Ave Score for Epoch:  1348.08
    
     === === === Iteration 128 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.01  Minutes
    
    Memories Elapsed:  858333
    Current Epsilon:  0.05
    Ave Score for Epoch:  1702.04
    
     === === === Iteration 129 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.91  Minutes
    
    Memories Elapsed:  865951
    Current Epsilon:  0.05
    Ave Score for Epoch:  1478.94
    
     === === === Iteration 130 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.83  Minutes
    
    Memories Elapsed:  873774
    Current Epsilon:  0.05
    Ave Score for Epoch:  1594.04
    
     === === === Iteration 131 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.75  Minutes
    
    Memories Elapsed:  881576
    Current Epsilon:  0.05
    Ave Score for Epoch:  1559.59
    
     === === === Iteration 132 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.65  Minutes
    
    Memories Elapsed:  889928
    Current Epsilon:  0.05
    Ave Score for Epoch:  1685.06
    
     === === === Iteration 133 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.56  Minutes
    
    Memories Elapsed:  897920
    Current Epsilon:  0.05
    Ave Score for Epoch:  1667.18
    
     === === === Iteration 134 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.47  Minutes
    
    Memories Elapsed:  907039
    Current Epsilon:  0.05
    Ave Score for Epoch:  1955.35
    
     === === === Iteration 135 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.39  Minutes
    
    Memories Elapsed:  916216
    Current Epsilon:  0.04
    Ave Score for Epoch:  1924.73
    
     === === === Iteration 136 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.29  Minutes
    
    Memories Elapsed:  923851
    Current Epsilon:  0.04
    Ave Score for Epoch:  1539.84
    
     === === === Iteration 137 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.18  Minutes
    
    Memories Elapsed:  931990
    Current Epsilon:  0.04
    Ave Score for Epoch:  1686.45
    
     === === === Iteration 138 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.07  Minutes
    
    Memories Elapsed:  941297
    Current Epsilon:  0.04
    Ave Score for Epoch:  1986.37
    
     === === === Iteration 139 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.96  Minutes
    
    Memories Elapsed:  949748
    Current Epsilon:  0.04
    Ave Score for Epoch:  1753.88
    
     === === === Iteration 140 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.87  Minutes
    
    Memories Elapsed:  958847
    Current Epsilon:  0.04
    Ave Score for Epoch:  1971.76
    
     === === === Iteration 141 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.78  Minutes
    
    Memories Elapsed:  968567
    Current Epsilon:  0.04
    Ave Score for Epoch:  2113.39
    
     === === === Iteration 142 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.67  Minutes
    
    Memories Elapsed:  976788
    Current Epsilon:  0.04
    Ave Score for Epoch:  1724.98
    
     === === === Iteration 143 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.54  Minutes
    
    Memories Elapsed:  984700
    Current Epsilon:  0.04
    Ave Score for Epoch:  1634.04
    
     === === === Iteration 144 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.42  Minutes
    
    Memories Elapsed:  993155
    Current Epsilon:  0.04
    Ave Score for Epoch:  1740.24
    
     === === === Iteration 145 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.31  Minutes
    
    Memories Elapsed:  1003091
    Current Epsilon:  0.04
    Ave Score for Epoch:  2148.16
    
     === === === Iteration 146 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.19  Minutes
    
    Memories Elapsed:  1011258
    Current Epsilon:  0.03
    Ave Score for Epoch:  1686.12
    
     === === === Iteration 147 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.06  Minutes
    
    Memories Elapsed:  1019079
    Current Epsilon:  0.03
    Ave Score for Epoch:  1593.96
    
     === === === Iteration 148 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.94  Minutes
    
    Memories Elapsed:  1027715
    Current Epsilon:  0.03
    Ave Score for Epoch:  1844.73
    
     === === === Iteration 149 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.82  Minutes
    
    Memories Elapsed:  1036163
    Current Epsilon:  0.03
    Ave Score for Epoch:  1767.92
    
     === === === Iteration 150 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.69  Minutes
    
    Memories Elapsed:  1044916
    Current Epsilon:  0.03
    Ave Score for Epoch:  1841.88
    
     === === === Iteration 151 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.57  Minutes
    
    Memories Elapsed:  1052815
    Current Epsilon:  0.03
    Ave Score for Epoch:  1632.00
    
     === === === Iteration 152 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.44  Minutes
    
    Memories Elapsed:  1060671
    Current Epsilon:  0.03
    Ave Score for Epoch:  1556.16
    
     === === === Iteration 153 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.31  Minutes
    
    Memories Elapsed:  1069041
    Current Epsilon:  0.03
    Ave Score for Epoch:  1721.22
    
     === === === Iteration 154 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.19  Minutes
    
    Memories Elapsed:  1077845
    Current Epsilon:  0.03
    Ave Score for Epoch:  1852.90
    
     === === === Iteration 155 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.06  Minutes
    
    Memories Elapsed:  1086269
    Current Epsilon:  0.03
    Ave Score for Epoch:  1709.06
    
     === === === Iteration 156 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.94  Minutes
    
    Memories Elapsed:  1096837
    Current Epsilon:  0.03
    Ave Score for Epoch:  2341.96
    
     === === === Iteration 157 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.82  Minutes
    
    Memories Elapsed:  1106723
    Current Epsilon:  0.03
    Ave Score for Epoch:  2168.00
    
     === === === Iteration 158 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.69  Minutes
    
    Memories Elapsed:  1116358
    Current Epsilon:  0.03
    Ave Score for Epoch:  2084.41
    
     === === === Iteration 159 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.57  Minutes
    
    Memories Elapsed:  1126134
    Current Epsilon:  0.03
    Ave Score for Epoch:  2144.08
    
     === === === Iteration 160 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.45  Minutes
    
    Memories Elapsed:  1136630
    Current Epsilon:  0.03
    Ave Score for Epoch:  2336.41
    
     === === === Iteration 161 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.32  Minutes
    
    Memories Elapsed:  1145501
    Current Epsilon:  0.02
    Ave Score for Epoch:  1906.29
    
     === === === Iteration 162 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.20  Minutes
    
    Memories Elapsed:  1156122
    Current Epsilon:  0.02
    Ave Score for Epoch:  2340.57
    
     === === === Iteration 163 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.08  Minutes
    
    Memories Elapsed:  1165282
    Current Epsilon:  0.02
    Ave Score for Epoch:  1999.02
    
     === === === Iteration 164 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.96  Minutes
    
    Memories Elapsed:  1174471
    Current Epsilon:  0.02
    Ave Score for Epoch:  1925.47
    
     === === === Iteration 165 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.82  Minutes
    
    Memories Elapsed:  1183361
    Current Epsilon:  0.02
    Ave Score for Epoch:  1890.45
    
     === === === Iteration 166 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.69  Minutes
    
    Memories Elapsed:  1192356
    Current Epsilon:  0.02
    Ave Score for Epoch:  1895.67
    
     === === === Iteration 167 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.57  Minutes
    
    Memories Elapsed:  1203120
    Current Epsilon:  0.02
    Ave Score for Epoch:  2372.98
    
     === === === Iteration 168 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.44  Minutes
    
    Memories Elapsed:  1214247
    Current Epsilon:  0.02
    Ave Score for Epoch:  2556.41
    
     === === === Iteration 169 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.31  Minutes
    
    Memories Elapsed:  1224767
    Current Epsilon:  0.02
    Ave Score for Epoch:  2368.65
    
     === === === Iteration 170 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.18  Minutes
    
    Memories Elapsed:  1234525
    Current Epsilon:  0.02
    Ave Score for Epoch:  2158.04
    
     === === === Iteration 171 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.06  Minutes
    
    Memories Elapsed:  1244523
    Current Epsilon:  0.02
    Ave Score for Epoch:  2171.51
    
     === === === Iteration 172 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.95  Minutes
    
    Memories Elapsed:  1255055
    Current Epsilon:  0.02
    Ave Score for Epoch:  2353.14
    
     === === === Iteration 173 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.82  Minutes
    
    Memories Elapsed:  1265238
    Current Epsilon:  0.02
    Ave Score for Epoch:  2343.02
    
     === === === Iteration 174 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.70  Minutes
    
    Memories Elapsed:  1275475
    Current Epsilon:  0.02
    Ave Score for Epoch:  2264.16
    
     === === === Iteration 175 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.57  Minutes
    
    Memories Elapsed:  1285267
    Current Epsilon:  0.02
    Ave Score for Epoch:  2106.45
    
     === === === Iteration 176 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.44  Minutes
    
    Memories Elapsed:  1295566
    Current Epsilon:  0.02
    Ave Score for Epoch:  2238.94
    
     === === === Iteration 177 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.31  Minutes
    
    Memories Elapsed:  1305397
    Current Epsilon:  0.02
    Ave Score for Epoch:  2104.08
    
     === === === Iteration 178 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.18  Minutes
    
    Memories Elapsed:  1315784
    Current Epsilon:  0.02
    Ave Score for Epoch:  2309.14
    
     === === === Iteration 179 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.05  Minutes
    
    Memories Elapsed:  1326112
    Current Epsilon:  0.02
    Ave Score for Epoch:  2297.88
    
     === === === Iteration 180 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.91  Minutes
    
    Memories Elapsed:  1336327
    Current Epsilon:  0.02
    Ave Score for Epoch:  2294.45
    
     === === === Iteration 181 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.77  Minutes
    
    Memories Elapsed:  1346511
    Current Epsilon:  0.02
    Ave Score for Epoch:  2242.29
    
     === === === Iteration 182 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.64  Minutes
    
    Memories Elapsed:  1357521
    Current Epsilon:  0.02
    Ave Score for Epoch:  2574.94
    
     === === === Iteration 183 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.50  Minutes
    
    Memories Elapsed:  1366658
    Current Epsilon:  0.01
    Ave Score for Epoch:  1949.63
    
     === === === Iteration 184 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.36  Minutes
    
    Memories Elapsed:  1376703
    Current Epsilon:  0.01
    Ave Score for Epoch:  2211.59
    
     === === === Iteration 185 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.22  Minutes
    
    Memories Elapsed:  1385340
    Current Epsilon:  0.01
    Ave Score for Epoch:  1804.65
    
     === === === Iteration 186 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.08  Minutes
    
    Memories Elapsed:  1396399
    Current Epsilon:  0.01
    Ave Score for Epoch:  2496.90
    
     === === === Iteration 187 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.94  Minutes
    
    Memories Elapsed:  1406188
    Current Epsilon:  0.01
    Ave Score for Epoch:  2206.94
    
     === === === Iteration 188 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.79  Minutes
    
    Memories Elapsed:  1415520
    Current Epsilon:  0.01
    Ave Score for Epoch:  2018.20
    
     === === === Iteration 189 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.65  Minutes
    
    Memories Elapsed:  1426502
    Current Epsilon:  0.01
    Ave Score for Epoch:  2491.59
    
     === === === Iteration 190 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.50  Minutes
    
    Memories Elapsed:  1436163
    Current Epsilon:  0.01
    Ave Score for Epoch:  2120.24
    
     === === === Iteration 191 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.36  Minutes
    
    Memories Elapsed:  1446269
    Current Epsilon:  0.01
    Ave Score for Epoch:  2265.22
    
     === === === Iteration 192 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.21  Minutes
    
    Memories Elapsed:  1455706
    Current Epsilon:  0.01
    Ave Score for Epoch:  2037.80
    
     === === === Iteration 193 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.06  Minutes
    
    Memories Elapsed:  1465500
    Current Epsilon:  0.01
    Ave Score for Epoch:  2126.78
    
     === === === Iteration 194 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.91  Minutes
    
    Memories Elapsed:  1474471
    Current Epsilon:  0.01
    Ave Score for Epoch:  1875.02
    
     === === === Iteration 195 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.76  Minutes
    
    Memories Elapsed:  1485172
    Current Epsilon:  0.01
    Ave Score for Epoch:  2380.00
    
     === === === Iteration 196 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.61  Minutes
    
    Memories Elapsed:  1495889
    Current Epsilon:  0.01
    Ave Score for Epoch:  2379.67
    
     === === === Iteration 197 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.46  Minutes
    
    Memories Elapsed:  1506051
    Current Epsilon:  0.01
    Ave Score for Epoch:  2258.37
    
     === === === Iteration 198 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.30  Minutes
    
    Memories Elapsed:  1516994
    Current Epsilon:  0.01
    Ave Score for Epoch:  2440.49
    
     === === === Iteration 199 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.15  Minutes
    
    Memories Elapsed:  1527794
    Current Epsilon:  0.01
    Ave Score for Epoch:  2450.37
    


```python
def getMovingAve(movingAveSize, scores):
    movingAve = []
    
    for i in range(len(scores) // movingAveSize):
        if (i + 1) * movingAveSize < len(scores):
            movingAve.append(np.mean(scores[i*movingAveSize:(i+1)*movingAveSize]))
    return movingAve


def mapScores():
    movingAveSize = 100
    
    movingAveA = getMovingAve(movingAveSize, gameScoreLog)
    movingAveB = getMovingAve(movingAveSize//5, greedyScoreLog)
    
    print("Minimum score:   ", np.min(movingAveA))
    print("25th percentile: ", np.percentile(movingAveA,25))
    print("50th percentile: ", np.mean(movingAveA))
    print("75th percentile: ", np.percentile(movingAveA,75))
    print("Maximum score:   ", np.max(movingAveA),"\n")
    
    print("Minimum score:   ", np.min(movingAveB))
    print("25th percentile: ", np.percentile(movingAveB,25))
    print("50th percentile: ", np.mean(movingAveB))
    print("75th percentile: ", np.percentile(movingAveB,75))
    print("Maximum score:   ", np.max(movingAveB),"\n")
    
    plt.figure(figsize=(10,6))
    plt.scatter(np.arange(len(movingAveA)),movingAveA)
    plt.scatter(np.arange(len(movingAveB)),movingAveB)
    plt.plot(np.arange(len(movingAveA)),np.ones(len(movingAveA))*np.mean(movingAveA))
    
mapScores()
```

    Minimum score:    999.0
    25th percentile:  1185.3400000000001
    50th percentile:  1538.30101010101
    75th percentile:  1844.32
    Maximum score:    2461.2 
    
    Minimum score:    766.8
    25th percentile:  1797.1
    50th percentile:  1973.337373737374
    75th percentile:  2252.3999999999996
    Maximum score:    2752.0 
    
    


![png](output_29_1.png)



```python
def testAgent():
    games = 1000
    
    scores = np.zeros(games)
    
    for i in range(games):
        scores[i] = (run(initializeEnv(),agent,memCount,True))
    
    print("Minimum score:   ", np.min(scores))
    print("25th percentile: ", np.percentile(scores,25))
    print("50th percentile: ", np.mean(scores))
    print("75th percentile: ", np.percentile(scores,75))
    print("Maximum score:   ", np.max(scores))
    

testAgent()
```


```python

```
