# Frozen Lake Game
Markov Decision Process AI using C++ to solve the frozen lake game. You can also play the frozen lake game yourself! 

To play the game:
1. Install python3-numpy 
2. python3 play_game.py --map maps/bridge.json

To solve the game:
1. Install python3-numpy  
2. Environment Setup:  
   cd frozen_lake   
   mkdir build  
   cd build  
   cmake ../  
   make  
   cd src

3. Value Iteration (Parameters are variable):  
   ./frozen_lake --agent v --map ../maps/bridge.json --epsilon 0.99 --alpha 1.0 --iteration 500 --gamma 0.99  

4. Policy Iteration (Parameters are variable):    
   ./frozen_lake --agent p --map ../maps/bridge_stochastic.json --epsilon 0.2  --alpha 0.05 --iteration 500 --gamma 0.99  

5. Q-Learning (Parameters are variable):  
   ./frozen_lake --agent q --map ../maps/cliff_stochastic.json --epsilon 0.2  --alpha 0.2 --iteration 100 --gamma 0.99