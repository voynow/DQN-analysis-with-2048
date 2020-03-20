
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
    
    if rand < 1 - EPSILON_RATE * iteration:
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
    print("Current Epsilon: ",format(1 - EPSILON_RATE * iteration,'.4f'))
    print("Ave Score for Epoch: ", format(np.mean(scoreLog[len(scoreLog) - EPISODES : len(scoreLog)-1]) ,'.2f'))
```


```python
def printTrainInfo():
    print("Maximum memory size:",MAX_MEM_SIZE)
    print("Sample size for training:",SAMPLE_SIZE)
    print("Batch size from sample:",BATCH_SIZE)
    print("Epslon Rate:",format(EPSILON_RATE,'.4f'))
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

    Using TensorFlow backend.
    


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

    WARNING: Logging before flag parsing goes to stderr.
    W0320 12:07:25.630152 17840 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0320 12:07:25.687612 17840 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0320 12:07:25.708161 17840 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0320 12:07:25.747581 17840 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.
    
    W0320 12:07:25.765968 17840 deprecation.py:506] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    W0320 12:07:25.890716 17840 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    


```python
ITERATIONS = 200
EPISODES = 50
TESTS = EPISODES // 5
EPOCHS = 4
MIN_EPSILON = 0.01
EPSILON_RATE = (1 - MIN_EPSILON) / ITERATIONS
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

    W0320 12:07:26.021060 17840 deprecation_wrapper.py:119] From C:\ProgramData\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:2741: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    
    

    Maximum memory size: 200000
    Sample size for training: 1000
    Batch size from sample: 10
    Epslon Rate: 0.0049
    
     === === === Iteration 0 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  36.48  Minutes
    
    Memories Elapsed:  6255
    Current Epsilon:  1.0000
    Ave Score for Epoch:  1171.51
    
     === === === Iteration 1 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  33.74  Minutes
    
    Memories Elapsed:  12676
    Current Epsilon:  0.9950
    Ave Score for Epoch:  1265.96
    
     === === === Iteration 2 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  30.65  Minutes
    
    Memories Elapsed:  18888
    Current Epsilon:  0.9901
    Ave Score for Epoch:  1188.49
    
     === === === Iteration 3 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  29.51  Minutes
    
    Memories Elapsed:  24613
    Current Epsilon:  0.9851
    Ave Score for Epoch:  1055.92
    
     === === === Iteration 4 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  28.23  Minutes
    
    Memories Elapsed:  30610
    Current Epsilon:  0.9802
    Ave Score for Epoch:  1135.67
    
     === === === Iteration 5 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  27.68  Minutes
    
    Memories Elapsed:  37003
    Current Epsilon:  0.9752
    Ave Score for Epoch:  1253.63
    
     === === === Iteration 6 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  26.95  Minutes
    
    Memories Elapsed:  43457
    Current Epsilon:  0.9703
    Ave Score for Epoch:  1258.29
    
     === === === Iteration 7 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  27.11  Minutes
    
    Memories Elapsed:  49651
    Current Epsilon:  0.9654
    Ave Score for Epoch:  1165.71
    
     === === === Iteration 8 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  27.65  Minutes
    
    Memories Elapsed:  55509
    Current Epsilon:  0.9604
    Ave Score for Epoch:  1073.80
    
     === === === Iteration 9 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  27.52  Minutes
    
    Memories Elapsed:  61439
    Current Epsilon:  0.9555
    Ave Score for Epoch:  1067.59
    
     === === === Iteration 10 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  27.20  Minutes
    
    Memories Elapsed:  67578
    Current Epsilon:  0.9505
    Ave Score for Epoch:  1151.67
    
     === === === Iteration 11 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  26.65  Minutes
    
    Memories Elapsed:  73483
    Current Epsilon:  0.9456
    Ave Score for Epoch:  1129.80
    
     === === === Iteration 12 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  25.96  Minutes
    
    Memories Elapsed:  79688
    Current Epsilon:  0.9406
    Ave Score for Epoch:  1171.18
    
     === === === Iteration 13 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  25.53  Minutes
    
    Memories Elapsed:  86123
    Current Epsilon:  0.9356
    Ave Score for Epoch:  1222.86
    
     === === === Iteration 14 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  25.21  Minutes
    
    Memories Elapsed:  92431
    Current Epsilon:  0.9307
    Ave Score for Epoch:  1198.12
    
     === === === Iteration 15 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  24.83  Minutes
    
    Memories Elapsed:  98148
    Current Epsilon:  0.9257
    Ave Score for Epoch:  1031.35
    
     === === === Iteration 16 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  24.40  Minutes
    
    Memories Elapsed:  103994
    Current Epsilon:  0.9208
    Ave Score for Epoch:  1074.20
    
     === === === Iteration 17 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  24.06  Minutes
    
    Memories Elapsed:  109993
    Current Epsilon:  0.9159
    Ave Score for Epoch:  1137.31
    
     === === === Iteration 18 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.89  Minutes
    
    Memories Elapsed:  116593
    Current Epsilon:  0.9109
    Ave Score for Epoch:  1284.82
    
     === === === Iteration 19 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.56  Minutes
    
    Memories Elapsed:  122902
    Current Epsilon:  0.9060
    Ave Score for Epoch:  1200.16
    
     === === === Iteration 20 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.35  Minutes
    
    Memories Elapsed:  128523
    Current Epsilon:  0.9010
    Ave Score for Epoch:  1010.86
    
     === === === Iteration 21 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.39  Minutes
    
    Memories Elapsed:  134783
    Current Epsilon:  0.8961
    Ave Score for Epoch:  1172.65
    
     === === === Iteration 22 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.30  Minutes
    
    Memories Elapsed:  140984
    Current Epsilon:  0.8911
    Ave Score for Epoch:  1194.78
    
     === === === Iteration 23 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  23.13  Minutes
    
    Memories Elapsed:  147157
    Current Epsilon:  0.8861
    Ave Score for Epoch:  1158.12
    
     === === === Iteration 24 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  22.88  Minutes
    
    Memories Elapsed:  152964
    Current Epsilon:  0.8812
    Ave Score for Epoch:  1065.88
    
     === === === Iteration 25 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  22.55  Minutes
    
    Memories Elapsed:  158940
    Current Epsilon:  0.8762
    Ave Score for Epoch:  1127.84
    
     === === === Iteration 26 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  22.31  Minutes
    
    Memories Elapsed:  165125
    Current Epsilon:  0.8713
    Ave Score for Epoch:  1163.76
    
     === === === Iteration 27 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  22.11  Minutes
    
    Memories Elapsed:  171388
    Current Epsilon:  0.8663
    Ave Score for Epoch:  1194.53
    
     === === === Iteration 28 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.93  Minutes
    
    Memories Elapsed:  177785
    Current Epsilon:  0.8614
    Ave Score for Epoch:  1233.31
    
     === === === Iteration 29 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.76  Minutes
    
    Memories Elapsed:  183861
    Current Epsilon:  0.8565
    Ave Score for Epoch:  1160.98
    
     === === === Iteration 30 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.57  Minutes
    
    Memories Elapsed:  189505
    Current Epsilon:  0.8515
    Ave Score for Epoch:  1044.08
    
     === === === Iteration 31 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.40  Minutes
    
    Memories Elapsed:  195689
    Current Epsilon:  0.8466
    Ave Score for Epoch:  1167.27
    
     === === === Iteration 32 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.20  Minutes
    
    Memories Elapsed:  202220
    Current Epsilon:  0.8416
    Ave Score for Epoch:  1279.18
    
     === === === Iteration 33 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  21.00  Minutes
    
    Memories Elapsed:  208422
    Current Epsilon:  0.8367
    Ave Score for Epoch:  1175.51
    
     === === === Iteration 34 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.74  Minutes
    
    Memories Elapsed:  214318
    Current Epsilon:  0.8317
    Ave Score for Epoch:  1108.73
    
     === === === Iteration 35 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.51  Minutes
    
    Memories Elapsed:  220157
    Current Epsilon:  0.8267
    Ave Score for Epoch:  1064.90
    
     === === === Iteration 36 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.32  Minutes
    
    Memories Elapsed:  226230
    Current Epsilon:  0.8218
    Ave Score for Epoch:  1119.27
    
     === === === Iteration 37 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.21  Minutes
    
    Memories Elapsed:  232144
    Current Epsilon:  0.8169
    Ave Score for Epoch:  1097.06
    
     === === === Iteration 38 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  20.04  Minutes
    
    Memories Elapsed:  238055
    Current Epsilon:  0.8119
    Ave Score for Epoch:  1096.73
    
     === === === Iteration 39 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.88  Minutes
    
    Memories Elapsed:  244088
    Current Epsilon:  0.8070
    Ave Score for Epoch:  1146.45
    
     === === === Iteration 40 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.71  Minutes
    
    Memories Elapsed:  250378
    Current Epsilon:  0.8020
    Ave Score for Epoch:  1172.98
    
     === === === Iteration 41 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.59  Minutes
    
    Memories Elapsed:  256966
    Current Epsilon:  0.7971
    Ave Score for Epoch:  1260.73
    
     === === === Iteration 42 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.62  Minutes
    
    Memories Elapsed:  263054
    Current Epsilon:  0.7921
    Ave Score for Epoch:  1161.39
    
     === === === Iteration 43 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.51  Minutes
    
    Memories Elapsed:  269515
    Current Epsilon:  0.7872
    Ave Score for Epoch:  1232.90
    
     === === === Iteration 44 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.57  Minutes
    
    Memories Elapsed:  276010
    Current Epsilon:  0.7822
    Ave Score for Epoch:  1256.90
    
     === === === Iteration 45 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.57  Minutes
    
    Memories Elapsed:  281896
    Current Epsilon:  0.7772
    Ave Score for Epoch:  1075.43
    
     === === === Iteration 46 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.54  Minutes
    
    Memories Elapsed:  288103
    Current Epsilon:  0.7723
    Ave Score for Epoch:  1196.08
    
     === === === Iteration 47 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.49  Minutes
    
    Memories Elapsed:  294367
    Current Epsilon:  0.7673
    Ave Score for Epoch:  1194.37
    
     === === === Iteration 48 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.40  Minutes
    
    Memories Elapsed:  300471
    Current Epsilon:  0.7624
    Ave Score for Epoch:  1148.65
    
     === === === Iteration 49 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.34  Minutes
    
    Memories Elapsed:  306382
    Current Epsilon:  0.7574
    Ave Score for Epoch:  1105.06
    
     === === === Iteration 50 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.26  Minutes
    
    Memories Elapsed:  312575
    Current Epsilon:  0.7525
    Ave Score for Epoch:  1186.04
    
     === === === Iteration 51 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.21  Minutes
    
    Memories Elapsed:  318449
    Current Epsilon:  0.7476
    Ave Score for Epoch:  1064.41
    
     === === === Iteration 52 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.30  Minutes
    
    Memories Elapsed:  324847
    Current Epsilon:  0.7426
    Ave Score for Epoch:  1235.35
    
     === === === Iteration 53 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.42  Minutes
    
    Memories Elapsed:  331332
    Current Epsilon:  0.7377
    Ave Score for Epoch:  1258.20
    
     === === === Iteration 54 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.50  Minutes
    
    Memories Elapsed:  337282
    Current Epsilon:  0.7327
    Ave Score for Epoch:  1100.57
    
     === === === Iteration 55 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.51  Minutes
    
    Memories Elapsed:  342957
    Current Epsilon:  0.7278
    Ave Score for Epoch:  1016.49
    
     === === === Iteration 56 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.61  Minutes
    
    Memories Elapsed:  348945
    Current Epsilon:  0.7228
    Ave Score for Epoch:  1096.49
    
     === === === Iteration 57 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.54  Minutes
    
    Memories Elapsed:  355453
    Current Epsilon:  0.7179
    Ave Score for Epoch:  1238.12
    
     === === === Iteration 58 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.54  Minutes
    
    Memories Elapsed:  361456
    Current Epsilon:  0.7129
    Ave Score for Epoch:  1133.96
    
     === === === Iteration 59 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.45  Minutes
    
    Memories Elapsed:  367480
    Current Epsilon:  0.7080
    Ave Score for Epoch:  1135.84
    
     === === === Iteration 60 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.45  Minutes
    
    Memories Elapsed:  373637
    Current Epsilon:  0.7030
    Ave Score for Epoch:  1179.43
    
     === === === Iteration 61 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.50  Minutes
    
    Memories Elapsed:  379758
    Current Epsilon:  0.6981
    Ave Score for Epoch:  1160.73
    
     === === === Iteration 62 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.50  Minutes
    
    Memories Elapsed:  385107
    Current Epsilon:  0.6931
    Ave Score for Epoch:  926.04
    
     === === === Iteration 63 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.55  Minutes
    
    Memories Elapsed:  391439
    Current Epsilon:  0.6882
    Ave Score for Epoch:  1208.98
    
     === === === Iteration 64 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.57  Minutes
    
    Memories Elapsed:  397217
    Current Epsilon:  0.6832
    Ave Score for Epoch:  1029.63
    
     === === === Iteration 65 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.60  Minutes
    
    Memories Elapsed:  403200
    Current Epsilon:  0.6783
    Ave Score for Epoch:  1125.63
    
     === === === Iteration 66 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.64  Minutes
    
    Memories Elapsed:  409263
    Current Epsilon:  0.6733
    Ave Score for Epoch:  1134.61
    
     === === === Iteration 67 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.48  Minutes
    
    Memories Elapsed:  415223
    Current Epsilon:  0.6683
    Ave Score for Epoch:  1116.41
    
     === === === Iteration 68 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.27  Minutes
    
    Memories Elapsed:  420851
    Current Epsilon:  0.6634
    Ave Score for Epoch:  1021.22
    
     === === === Iteration 69 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  19.11  Minutes
    
    Memories Elapsed:  426459
    Current Epsilon:  0.6584
    Ave Score for Epoch:  1005.31
    
     === === === Iteration 70 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.90  Minutes
    
    Memories Elapsed:  431857
    Current Epsilon:  0.6535
    Ave Score for Epoch:  966.12
    
     === === === Iteration 71 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.69  Minutes
    
    Memories Elapsed:  438315
    Current Epsilon:  0.6485
    Ave Score for Epoch:  1262.37
    
     === === === Iteration 72 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.51  Minutes
    
    Memories Elapsed:  444361
    Current Epsilon:  0.6436
    Ave Score for Epoch:  1126.20
    
     === === === Iteration 73 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.30  Minutes
    
    Memories Elapsed:  450778
    Current Epsilon:  0.6387
    Ave Score for Epoch:  1257.31
    
     === === === Iteration 74 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  18.08  Minutes
    
    Memories Elapsed:  456360
    Current Epsilon:  0.6337
    Ave Score for Epoch:  1016.41
    
     === === === Iteration 75 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.84  Minutes
    
    Memories Elapsed:  461974
    Current Epsilon:  0.6288
    Ave Score for Epoch:  1010.45
    
     === === === Iteration 76 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.66  Minutes
    
    Memories Elapsed:  468203
    Current Epsilon:  0.6238
    Ave Score for Epoch:  1191.84
    
     === === === Iteration 77 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.50  Minutes
    
    Memories Elapsed:  474096
    Current Epsilon:  0.6189
    Ave Score for Epoch:  1088.57
    
     === === === Iteration 78 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.32  Minutes
    
    Memories Elapsed:  480012
    Current Epsilon:  0.6139
    Ave Score for Epoch:  1055.59
    
     === === === Iteration 79 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  17.17  Minutes
    
    Memories Elapsed:  485938
    Current Epsilon:  0.6090
    Ave Score for Epoch:  1102.12
    
     === === === Iteration 80 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.98  Minutes
    
    Memories Elapsed:  491775
    Current Epsilon:  0.6040
    Ave Score for Epoch:  1049.14
    
     === === === Iteration 81 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.82  Minutes
    
    Memories Elapsed:  498380
    Current Epsilon:  0.5991
    Ave Score for Epoch:  1282.12
    
     === === === Iteration 82 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.64  Minutes
    
    Memories Elapsed:  504519
    Current Epsilon:  0.5941
    Ave Score for Epoch:  1131.51
    
     === === === Iteration 83 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.45  Minutes
    
    Memories Elapsed:  510673
    Current Epsilon:  0.5892
    Ave Score for Epoch:  1154.78
    
     === === === Iteration 84 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.27  Minutes
    
    Memories Elapsed:  516833
    Current Epsilon:  0.5842
    Ave Score for Epoch:  1163.51
    
     === === === Iteration 85 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  16.09  Minutes
    
    Memories Elapsed:  523145
    Current Epsilon:  0.5793
    Ave Score for Epoch:  1192.49
    
     === === === Iteration 86 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.91  Minutes
    
    Memories Elapsed:  529516
    Current Epsilon:  0.5743
    Ave Score for Epoch:  1216.16
    
     === === === Iteration 87 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.73  Minutes
    
    Memories Elapsed:  535627
    Current Epsilon:  0.5694
    Ave Score for Epoch:  1132.65
    
     === === === Iteration 88 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.55  Minutes
    
    Memories Elapsed:  541610
    Current Epsilon:  0.5644
    Ave Score for Epoch:  1100.49
    
     === === === Iteration 89 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.36  Minutes
    
    Memories Elapsed:  547657
    Current Epsilon:  0.5595
    Ave Score for Epoch:  1121.71
    
     === === === Iteration 90 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.19  Minutes
    
    Memories Elapsed:  553673
    Current Epsilon:  0.5545
    Ave Score for Epoch:  1117.47
    
     === === === Iteration 91 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  15.01  Minutes
    
    Memories Elapsed:  560085
    Current Epsilon:  0.5495
    Ave Score for Epoch:  1237.71
    
     === === === Iteration 92 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.83  Minutes
    
    Memories Elapsed:  566043
    Current Epsilon:  0.5446
    Ave Score for Epoch:  1115.10
    
     === === === Iteration 93 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.66  Minutes
    
    Memories Elapsed:  571728
    Current Epsilon:  0.5396
    Ave Score for Epoch:  1036.41
    
     === === === Iteration 94 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.49  Minutes
    
    Memories Elapsed:  577510
    Current Epsilon:  0.5347
    Ave Score for Epoch:  1041.55
    
     === === === Iteration 95 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.33  Minutes
    
    Memories Elapsed:  583950
    Current Epsilon:  0.5298
    Ave Score for Epoch:  1241.39
    
     === === === Iteration 96 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.17  Minutes
    
    Memories Elapsed:  590094
    Current Epsilon:  0.5248
    Ave Score for Epoch:  1144.00
    
     === === === Iteration 97 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  14.00  Minutes
    
    Memories Elapsed:  595831
    Current Epsilon:  0.5199
    Ave Score for Epoch:  1036.49
    
     === === === Iteration 98 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.83  Minutes
    
    Memories Elapsed:  601813
    Current Epsilon:  0.5149
    Ave Score for Epoch:  1101.63
    
     === === === Iteration 99 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.67  Minutes
    
    Memories Elapsed:  607891
    Current Epsilon:  0.5100
    Ave Score for Epoch:  1104.57
    
     === === === Iteration 100 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.53  Minutes
    
    Memories Elapsed:  614720
    Current Epsilon:  0.5050
    Ave Score for Epoch:  1342.04
    
     === === === Iteration 101 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.37  Minutes
    
    Memories Elapsed:  621106
    Current Epsilon:  0.5001
    Ave Score for Epoch:  1206.86
    
     === === === Iteration 102 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.25  Minutes
    
    Memories Elapsed:  627170
    Current Epsilon:  0.4951
    Ave Score for Epoch:  1121.06
    
     === === === Iteration 103 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  13.13  Minutes
    
    Memories Elapsed:  633797
    Current Epsilon:  0.4902
    Ave Score for Epoch:  1289.55
    
     === === === Iteration 104 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.99  Minutes
    
    Memories Elapsed:  639592
    Current Epsilon:  0.4852
    Ave Score for Epoch:  1030.45
    
     === === === Iteration 105 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.86  Minutes
    
    Memories Elapsed:  646433
    Current Epsilon:  0.4803
    Ave Score for Epoch:  1327.43
    
     === === === Iteration 106 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.74  Minutes
    
    Memories Elapsed:  652892
    Current Epsilon:  0.4753
    Ave Score for Epoch:  1243.59
    
     === === === Iteration 107 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.61  Minutes
    
    Memories Elapsed:  659097
    Current Epsilon:  0.4704
    Ave Score for Epoch:  1149.63
    
     === === === Iteration 108 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.49  Minutes
    
    Memories Elapsed:  664903
    Current Epsilon:  0.4654
    Ave Score for Epoch:  1049.96
    
     === === === Iteration 109 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.35  Minutes
    
    Memories Elapsed:  670946
    Current Epsilon:  0.4605
    Ave Score for Epoch:  1139.51
    
     === === === Iteration 110 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.24  Minutes
    
    Memories Elapsed:  677498
    Current Epsilon:  0.4555
    Ave Score for Epoch:  1261.22
    
     === === === Iteration 111 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  12.10  Minutes
    
    Memories Elapsed:  683397
    Current Epsilon:  0.4506
    Ave Score for Epoch:  1054.04
    
     === === === Iteration 112 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.97  Minutes
    
    Memories Elapsed:  689447
    Current Epsilon:  0.4456
    Ave Score for Epoch:  1124.49
    
     === === === Iteration 113 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.84  Minutes
    
    Memories Elapsed:  695850
    Current Epsilon:  0.4407
    Ave Score for Epoch:  1228.90
    
     === === === Iteration 114 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.69  Minutes
    
    Memories Elapsed:  701938
    Current Epsilon:  0.4357
    Ave Score for Epoch:  1126.37
    
     === === === Iteration 115 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.54  Minutes
    
    Memories Elapsed:  708174
    Current Epsilon:  0.4308
    Ave Score for Epoch:  1168.00
    
     === === === Iteration 116 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.42  Minutes
    
    Memories Elapsed:  714557
    Current Epsilon:  0.4258
    Ave Score for Epoch:  1182.29
    
     === === === Iteration 117 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.29  Minutes
    
    Memories Elapsed:  720756
    Current Epsilon:  0.4209
    Ave Score for Epoch:  1153.06
    
     === === === Iteration 118 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.15  Minutes
    
    Memories Elapsed:  726896
    Current Epsilon:  0.4159
    Ave Score for Epoch:  1111.18
    
     === === === Iteration 119 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  11.02  Minutes
    
    Memories Elapsed:  733469
    Current Epsilon:  0.4110
    Ave Score for Epoch:  1267.43
    
     === === === Iteration 120 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.88  Minutes
    
    Memories Elapsed:  739490
    Current Epsilon:  0.4060
    Ave Score for Epoch:  1089.55
    
     === === === Iteration 121 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.75  Minutes
    
    Memories Elapsed:  746258
    Current Epsilon:  0.4011
    Ave Score for Epoch:  1294.86
    
     === === === Iteration 122 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.61  Minutes
    
    Memories Elapsed:  752869
    Current Epsilon:  0.3961
    Ave Score for Epoch:  1300.98
    
     === === === Iteration 123 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.47  Minutes
    
    Memories Elapsed:  758948
    Current Epsilon:  0.3912
    Ave Score for Epoch:  1121.80
    
     === === === Iteration 124 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.33  Minutes
    
    Memories Elapsed:  765167
    Current Epsilon:  0.3862
    Ave Score for Epoch:  1152.65
    
     === === === Iteration 125 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.22  Minutes
    
    Memories Elapsed:  771992
    Current Epsilon:  0.3813
    Ave Score for Epoch:  1305.14
    
     === === === Iteration 126 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  10.09  Minutes
    
    Memories Elapsed:  778946
    Current Epsilon:  0.3763
    Ave Score for Epoch:  1354.12
    
     === === === Iteration 127 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.94  Minutes
    
    Memories Elapsed:  785258
    Current Epsilon:  0.3714
    Ave Score for Epoch:  1194.69
    
     === === === Iteration 128 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.79  Minutes
    
    Memories Elapsed:  791962
    Current Epsilon:  0.3664
    Ave Score for Epoch:  1323.92
    
     === === === Iteration 129 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.66  Minutes
    
    Memories Elapsed:  798537
    Current Epsilon:  0.3615
    Ave Score for Epoch:  1244.24
    
     === === === Iteration 130 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.54  Minutes
    
    Memories Elapsed:  804946
    Current Epsilon:  0.3565
    Ave Score for Epoch:  1215.10
    
     === === === Iteration 131 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.43  Minutes
    
    Memories Elapsed:  811291
    Current Epsilon:  0.3516
    Ave Score for Epoch:  1198.20
    
     === === === Iteration 132 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.32  Minutes
    
    Memories Elapsed:  817933
    Current Epsilon:  0.3466
    Ave Score for Epoch:  1257.55
    
     === === === Iteration 133 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.21  Minutes
    
    Memories Elapsed:  824225
    Current Epsilon:  0.3417
    Ave Score for Epoch:  1170.29
    
     === === === Iteration 134 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  9.08  Minutes
    
    Memories Elapsed:  831181
    Current Epsilon:  0.3367
    Ave Score for Epoch:  1379.27
    
     === === === Iteration 135 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.96  Minutes
    
    Memories Elapsed:  837875
    Current Epsilon:  0.3318
    Ave Score for Epoch:  1310.20
    
     === === === Iteration 136 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.81  Minutes
    
    Memories Elapsed:  844763
    Current Epsilon:  0.3268
    Ave Score for Epoch:  1351.18
    
     === === === Iteration 137 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.66  Minutes
    
    Memories Elapsed:  851042
    Current Epsilon:  0.3219
    Ave Score for Epoch:  1168.90
    
     === === === Iteration 138 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.52  Minutes
    
    Memories Elapsed:  857700
    Current Epsilon:  0.3169
    Ave Score for Epoch:  1272.90
    
     === === === Iteration 139 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.38  Minutes
    
    Memories Elapsed:  864134
    Current Epsilon:  0.3120
    Ave Score for Epoch:  1209.14
    
     === === === Iteration 140 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.24  Minutes
    
    Memories Elapsed:  870852
    Current Epsilon:  0.3070
    Ave Score for Epoch:  1294.69
    
     === === === Iteration 141 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  8.11  Minutes
    
    Memories Elapsed:  877098
    Current Epsilon:  0.3021
    Ave Score for Epoch:  1197.22
    
     === === === Iteration 142 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.97  Minutes
    
    Memories Elapsed:  883436
    Current Epsilon:  0.2971
    Ave Score for Epoch:  1194.53
    
     === === === Iteration 143 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.85  Minutes
    
    Memories Elapsed:  889625
    Current Epsilon:  0.2922
    Ave Score for Epoch:  1159.35
    
     === === === Iteration 144 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.72  Minutes
    
    Memories Elapsed:  896293
    Current Epsilon:  0.2872
    Ave Score for Epoch:  1246.37
    
     === === === Iteration 145 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.59  Minutes
    
    Memories Elapsed:  902726
    Current Epsilon:  0.2823
    Ave Score for Epoch:  1216.57
    
     === === === Iteration 146 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.46  Minutes
    
    Memories Elapsed:  909215
    Current Epsilon:  0.2773
    Ave Score for Epoch:  1232.90
    
     === === === Iteration 147 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.33  Minutes
    
    Memories Elapsed:  915596
    Current Epsilon:  0.2724
    Ave Score for Epoch:  1152.41
    
     === === === Iteration 148 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.20  Minutes
    
    Memories Elapsed:  922163
    Current Epsilon:  0.2674
    Ave Score for Epoch:  1251.43
    
     === === === Iteration 149 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  7.07  Minutes
    
    Memories Elapsed:  928978
    Current Epsilon:  0.2625
    Ave Score for Epoch:  1292.98
    
     === === === Iteration 150 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.95  Minutes
    
    Memories Elapsed:  935741
    Current Epsilon:  0.2575
    Ave Score for Epoch:  1317.47
    
     === === === Iteration 151 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.81  Minutes
    
    Memories Elapsed:  942097
    Current Epsilon:  0.2526
    Ave Score for Epoch:  1213.14
    
     === === === Iteration 152 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.68  Minutes
    
    Memories Elapsed:  949020
    Current Epsilon:  0.2476
    Ave Score for Epoch:  1334.69
    
     === === === Iteration 153 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.54  Minutes
    
    Memories Elapsed:  956134
    Current Epsilon:  0.2427
    Ave Score for Epoch:  1366.29
    
     === === === Iteration 154 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.40  Minutes
    
    Memories Elapsed:  963118
    Current Epsilon:  0.2377
    Ave Score for Epoch:  1388.82
    
     === === === Iteration 155 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.26  Minutes
    
    Memories Elapsed:  970066
    Current Epsilon:  0.2328
    Ave Score for Epoch:  1342.45
    
     === === === Iteration 156 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.13  Minutes
    
    Memories Elapsed:  977111
    Current Epsilon:  0.2278
    Ave Score for Epoch:  1377.31
    
     === === === Iteration 157 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  6.00  Minutes
    
    Memories Elapsed:  984092
    Current Epsilon:  0.2229
    Ave Score for Epoch:  1373.63
    
     === === === Iteration 158 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.86  Minutes
    
    Memories Elapsed:  990795
    Current Epsilon:  0.2179
    Ave Score for Epoch:  1301.63
    
     === === === Iteration 159 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.73  Minutes
    
    Memories Elapsed:  998538
    Current Epsilon:  0.2130
    Ave Score for Epoch:  1596.65
    
     === === === Iteration 160 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.59  Minutes
    
    Memories Elapsed:  1005956
    Current Epsilon:  0.2080
    Ave Score for Epoch:  1499.84
    
     === === === Iteration 161 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.45  Minutes
    
    Memories Elapsed:  1013353
    Current Epsilon:  0.2031
    Ave Score for Epoch:  1480.73
    
     === === === Iteration 162 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.33  Minutes
    
    Memories Elapsed:  1020409
    Current Epsilon:  0.1981
    Ave Score for Epoch:  1405.39
    
     === === === Iteration 163 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.19  Minutes
    
    Memories Elapsed:  1027514
    Current Epsilon:  0.1932
    Ave Score for Epoch:  1393.55
    
     === === === Iteration 164 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  5.06  Minutes
    
    Memories Elapsed:  1034640
    Current Epsilon:  0.1882
    Ave Score for Epoch:  1408.73
    
     === === === Iteration 165 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.93  Minutes
    
    Memories Elapsed:  1042157
    Current Epsilon:  0.1833
    Ave Score for Epoch:  1506.29
    
     === === === Iteration 166 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.79  Minutes
    
    Memories Elapsed:  1049494
    Current Epsilon:  0.1783
    Ave Score for Epoch:  1452.41
    
     === === === Iteration 167 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.65  Minutes
    
    Memories Elapsed:  1056922
    Current Epsilon:  0.1734
    Ave Score for Epoch:  1487.18
    
     === === === Iteration 168 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.51  Minutes
    
    Memories Elapsed:  1064512
    Current Epsilon:  0.1684
    Ave Score for Epoch:  1514.20
    
     === === === Iteration 169 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.37  Minutes
    
    Memories Elapsed:  1072668
    Current Epsilon:  0.1635
    Ave Score for Epoch:  1719.02
    
     === === === Iteration 170 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.23  Minutes
    
    Memories Elapsed:  1079819
    Current Epsilon:  0.1585
    Ave Score for Epoch:  1397.80
    
     === === === Iteration 171 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  4.10  Minutes
    
    Memories Elapsed:  1086793
    Current Epsilon:  0.1536
    Ave Score for Epoch:  1392.16
    
     === === === Iteration 172 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.98  Minutes
    
    Memories Elapsed:  1093964
    Current Epsilon:  0.1486
    Ave Score for Epoch:  1378.45
    
     === === === Iteration 173 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.84  Minutes
    
    Memories Elapsed:  1101891
    Current Epsilon:  0.1437
    Ave Score for Epoch:  1594.29
    
     === === === Iteration 174 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.70  Minutes
    
    Memories Elapsed:  1109975
    Current Epsilon:  0.1387
    Ave Score for Epoch:  1678.20
    
     === === === Iteration 175 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.57  Minutes
    
    Memories Elapsed:  1117822
    Current Epsilon:  0.1338
    Ave Score for Epoch:  1596.41
    
     === === === Iteration 176 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.43  Minutes
    
    Memories Elapsed:  1126005
    Current Epsilon:  0.1288
    Ave Score for Epoch:  1676.41
    
     === === === Iteration 177 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.29  Minutes
    
    Memories Elapsed:  1133980
    Current Epsilon:  0.1239
    Ave Score for Epoch:  1652.24
    
     === === === Iteration 178 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.15  Minutes
    
    Memories Elapsed:  1142508
    Current Epsilon:  0.1189
    Ave Score for Epoch:  1782.61
    
     === === === Iteration 179 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  3.01  Minutes
    
    Memories Elapsed:  1150951
    Current Epsilon:  0.1140
    Ave Score for Epoch:  1808.41
    
     === === === Iteration 180 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.87  Minutes
    
    Memories Elapsed:  1159871
    Current Epsilon:  0.1090
    Ave Score for Epoch:  1946.69
    
     === === === Iteration 181 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.73  Minutes
    
    Memories Elapsed:  1168351
    Current Epsilon:  0.1041
    Ave Score for Epoch:  1786.45
    
     === === === Iteration 182 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.59  Minutes
    
    Memories Elapsed:  1177077
    Current Epsilon:  0.0991
    Ave Score for Epoch:  1848.90
    
     === === === Iteration 183 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.45  Minutes
    
    Memories Elapsed:  1185058
    Current Epsilon:  0.0942
    Ave Score for Epoch:  1615.59
    
     === === === Iteration 184 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.31  Minutes
    
    Memories Elapsed:  1194169
    Current Epsilon:  0.0892
    Ave Score for Epoch:  1992.41
    
     === === === Iteration 185 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.17  Minutes
    
    Memories Elapsed:  1202612
    Current Epsilon:  0.0843
    Ave Score for Epoch:  1713.14
    
     === === === Iteration 186 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  2.02  Minutes
    
    Memories Elapsed:  1211311
    Current Epsilon:  0.0793
    Ave Score for Epoch:  1838.37
    
     === === === Iteration 187 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.88  Minutes
    
    Memories Elapsed:  1219972
    Current Epsilon:  0.0744
    Ave Score for Epoch:  1845.14
    
     === === === Iteration 188 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.74  Minutes
    
    Memories Elapsed:  1228910
    Current Epsilon:  0.0694
    Ave Score for Epoch:  1824.41
    
     === === === Iteration 189 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.60  Minutes
    
    Memories Elapsed:  1238225
    Current Epsilon:  0.0645
    Ave Score for Epoch:  2032.00
    
     === === === Iteration 190 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.45  Minutes
    
    Memories Elapsed:  1247576
    Current Epsilon:  0.0595
    Ave Score for Epoch:  2016.82
    
     === === === Iteration 191 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.31  Minutes
    
    Memories Elapsed:  1257007
    Current Epsilon:  0.0546
    Ave Score for Epoch:  1988.00
    
     === === === Iteration 192 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.16  Minutes
    
    Memories Elapsed:  1266584
    Current Epsilon:  0.0496
    Ave Score for Epoch:  2136.82
    
     === === === Iteration 193 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  1.02  Minutes
    
    Memories Elapsed:  1276239
    Current Epsilon:  0.0447
    Ave Score for Epoch:  2144.33
    
     === === === Iteration 194 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.87  Minutes
    
    Memories Elapsed:  1285752
    Current Epsilon:  0.0397
    Ave Score for Epoch:  2058.45
    
     === === === Iteration 195 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.73  Minutes
    
    Memories Elapsed:  1295932
    Current Epsilon:  0.0348
    Ave Score for Epoch:  2319.18
    
     === === === Iteration 196 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.58  Minutes
    
    Memories Elapsed:  1306567
    Current Epsilon:  0.0298
    Ave Score for Epoch:  2416.57
    
     === === === Iteration 197 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.44  Minutes
    
    Memories Elapsed:  1317213
    Current Epsilon:  0.0249
    Ave Score for Epoch:  2392.90
    
     === === === Iteration 198 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.29  Minutes
    
    Memories Elapsed:  1328634
    Current Epsilon:  0.0199
    Ave Score for Epoch:  2632.16
    
     === === === Iteration 199 of 200 === === ===
    
    Training Model...
    
    Estimated time remaining:  0.15  Minutes
    
    Memories Elapsed:  1338223
    Current Epsilon:  0.0150
    Ave Score for Epoch:  2118.45
    


```python
def getMovingAve(movingAveSize, scores):
    movingAve = []
    
    for i in range(len(scores) // movingAveSize):
        if (i + 1) * movingAveSize < len(scores):
            movingAve.append(np.mean(scores[i*movingAveSize:(i+1)*movingAveSize]))
    return movingAve


def mapScores():
    movingAveSize = 50
    
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

    Minimum score:    932.08
    25th percentile:  1128.04
    50th percentile:  1295.8448241206029
    75th percentile:  1329.84
    Maximum score:    2629.92 
    
    Minimum score:    558.4
    25th percentile:  1256.8
    50th percentile:  1970.9728643216079
    75th percentile:  2560.0
    Maximum score:    3500.4 
    
    


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

    Minimum score:    436.0
    25th percentile:  1572.0
    50th percentile:  2502.936
    75th percentile:  3233.0
    Maximum score:    8008.0
    


```python

```
