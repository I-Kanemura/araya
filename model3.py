
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict , deque
import torch 
import torch.nn as nn
import torch.nn.functional as F
from perceiver_pytorch import Perceiver
from perceiver_pytorch import PerceiverIO

class GridWorld :
  def __init__( self , row = 3 , cal = 4 ) :
    self.action_move = [ (-1,0), (1,0), (0,-1), (0,1) ]
    self.action_list = [0,1,2,3]
    self.action_label = { 0 : 'up', 
                          1 : 'down',
                          2 : 'left',
                          3 : 'right' }
  
    self.row = row
    self.cal = cal
    self.start_state = ( 0 , 0 ) 
    self.goal_state = ( row-1 , cal-1 ) 
    self.agent_state = ( 0, 0 )

    self.wall_map = None
    self.reward_map = None
    self.create()

  @property
  def shape( self ) :
    return ( self.row , self.cal )
  
  def actions( self ) :
    return self.action_list

  # 迷路を作成
  def create( self ) :
    self.wall_map = np.zeros( ( self.row , self.cal ) ) 
    self.wall_map[1,1] = None
    self.reward_map = np.zeros( ( self.row , self.cal  ) ) 
    self.reward_map[2,3] = 5.0
    self.reward_map[1,3] = -5.0
  
  def next_state( self, state , action ) :
    move = self.action_move[action]
    next_state = ( state[0]+move[0], state[1]+move[1] )
    # 範囲外
    if next_state[0] < 0 or self.row <= next_state[0] or next_state[1] < 0 or self.cal <= next_state[1] :
      next_state = state
    # 壁激突
    elif self.wall_map[next_state] :
      next_state = state
    return next_state

  def get_reward( self, state, action, next_state ) :
    return self.reward_map[ next_state ]
  
  # 座標を1つずつ読み出す
  def states(self) :
    for i in range( self.row ) :
      for j in range( self.cal ) :
        yield( i , j )

  def reset( self ) :
    self.agent_state = ( 0, 0 )
    return self.agent_state
  
  def step( self , action ) :
    state = self.agent_state 
    move = self.action_move[action]
    next_state = ( state[0]+move[0], state[1]+move[1] )
    # 範囲外
    if next_state[0] < 0 or self.row <= next_state[0] or next_state[1] < 0 or self.cal <= next_state[1] :
      next_state = state
      reward = -1
    # 壁激突
    elif self.wall_map[next_state] :
      next_state = state
      reward = -1
    else :
      reward = self.get_reward( state , action , next_state )
    self.agent_state = next_state

    # ゴールはスキップ
    if self.agent_state == self.goal_state :
      done = True
    else :
      done = False

    return next_state, reward, done

    # 状態価値を書き出す  
  def printState( self , V ) :
    print_map = np.zeros( ( self.row , self.cal  ) )
    for i in range( self.row ) :
      for j in range( self.cal ) :
        print_map[  i , j ] = V[( i , j )] 
    return print_map
  
  def embedding( self, state ) :
    result = np.zeros((1 , 3,self.row, self.cal))
    '''
    map_list = [ (state[0],state[1]),
                  (state[0]-1,state[1]),
                  (state[0]+1,state[1]),
                  (state[0],state[1]-1),
                  (state[0],state[1]+1),
                  (state[0]+1,state[1]-1),
                  (state[0]+1,state[1]+1), 
                  (state[0]-1,state[1]-1),
                  (state[0]-1,state[1]+1)]
    '''
    map_list = [ (state[0],state[1]),
                  (state[0]-1,state[1]),
                  (state[0]+1,state[1]),
                  (state[0],state[1]-1),
                  (state[0],state[1]+1)]    
    
    for point in map_list :
      if point[0] < 0 or self.row <= point[0] or point[1] < 0 or self.cal <= point[1] :
        continue
      else :
        if self.wall_map[point] == None :
          result[0][0][point] = -1
        else :
          result[0][0][point] = 1

        result[0][1][point] = self.reward_map[point]  
        
    return np.copy( result.reshape([1 , 3, self.row*self.cal]) )


class NetModel( nn.Module ) :
  def __init__(self):
    super().__init__()
    self.Encoder = PerceiverIO(
      dim = 12,                    # dimension of sequence to be encoded
      queries_dim = 4,             # dimension of decoder queries
      logits_dim = 12,             # dimension of final logits
      depth = 6,                   # depth of net
      num_latents = 12,           # number of latents, or induced set points, or centroids. different papers giving it different names
      latent_dim = 4,            # latent dimension
      cross_heads = 1,             # number of heads for cross attention. paper said 1
      latent_heads = 8,            # number of heads for latent self attention, 8
      cross_dim_head = 64,         # number of dimensions per cross attention head
      latent_dim_head = 64,        # number of dimensions per latent self attention head
      weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
    )

  def forward( self, x ,action ):

    y = self.Encoder( torch.Tensor( x ), queries = torch.Tensor(action) ) # (1, 128, 100) - (batch, decoder seq, logits dim)
    
    return y

class Net( nn.Module ) :
  def __init__(self):
    super().__init__()
  
    self.e = 0.1
    self.gamma = 0.9
    self.lr = 0.01
    self.action_list = [0,1,2,3]
    self.action_move = [ (-1,0), (1,0), (0,-1), (0,1) ]
    self.action_size = len( self.action_list )
    self.PerceiverIONet = NetModel( )
    self.optimizer = torch.optim.SGD(self.PerceiverIONet.parameters(), lr=self.lr)


  def get_action( self , state ) :
    action = np.zeros((3,4))
    action_index = np.random.randint( 4 )
    action[state] = 1
    x = state[0]+self.action_move[action_index][0]
    if x < 0 or 2 < x :
      x = state[0]
    y = state[1]+self.action_move[action_index][1]
    if y < 0 or 3 < y :
      y = state[1]
    action[ x , y  ] = 1
    return action_index , np.copy( action )

  def update( self, state, action, reward, next_state, done ,flag=False ) :

    # 損失計算
    pred = self.PerceiverIONet( state , action )
    loss = torch.nn.functional.mse_loss( torch.Tensor(next_state), pred )

    # 学習
    if flag :
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    return loss.data

env = GridWorld()
agent = Net()


trial = 1000
reward = 0
history = []

for i in range( trial ) :
  state = env.reset()
  # 初期状態の空間情報
  state_map = env.embedding( state )

  total_loss = 0
  counter = 0
  done = False

  while not done :
    # 行動決定（ランダム）
    action_index , action = agent.get_action( state )
    # 動く前
    state_map = env.embedding( state )
  
    # 動いた先
    next_state, reward, done = env.step( action_index )
    # 空間情報を観測
    next_state_map = env.embedding( next_state )

  
    loss = agent.update( state_map , action , reward, next_state_map , done , True)
    

    total_loss += loss
    counter += 1
    state = next_state

  print( 'Ep:', i ,'Loss:', total_loss / counter )
  # 各試行ごとのロス変化
  history.append( total_loss / counter )


plt.plot( history )
plt.savefig('img_loss.pdf')


print( agent.PerceiverIONet.Encoder.latents )


trial = 100
history_loss = []
for i in range( trial ) :
  state = env.reset()
  # 初期状態の空間情報
  state_map = env.embedding( state )

  total_loss = 0
  total_reward = 0
  counter = 0
  done = False
  while not done :
    # 行動決定（ランダム）
    action_index , action = agent.get_action( state )
    # 動く前
    state_map = env.embedding( state )
  
    # 動いた先
    next_state, reward, done = env.step( action_index )
    # 空間情報を観測
    next_state_map = env.embedding( next_state )

  
    loss = agent.update( state_map , action , reward, next_state_map , done , False)
    
    total_loss += loss
    
    counter += 1
    state = next_state 

  print( 'Ep:', i ,'Loss:', total_loss / counter )
  # 各試行ごとのロス変化
  history_loss.append( total_loss / counter )


plt.plot( history_loss )
plt.savefig('img_loss_result.pdf')

