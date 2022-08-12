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
    self.reward_map[2,3] = 1.0
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
    result = np.zeros((1, self.row, self.cal ,3))
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
          result[0][point][0] = 0
        else :
          result[0][point][0] = 1

        result[0][point][1] = self.reward_map[point]  
        
    return np.copy( result )


class PerceiverQLNet( nn.Module ) :
  def __init__(self):
    super().__init__()
    # ロジストかけた結果が返ってくる
    # head = Multi-head Attentionのヘッド数
    self.Encoder = Perceiver(
                input_channels = 3,          # number of channels for each token of the input(単語数)
                input_axis = 2,              # number of axis for input data (2 for images, 3 for video)

                                             # フーリエ
                num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1) 6
                max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is 10.


                depth = 6,                   # depth of net. The shape of the final attention mechanism will be: 6
                                             # depth * (cross attention -> self_per_cross_attn * self attention)

                num_latents = 4,           # number of latents, or induced set points, or centroids. different papers giving it different names 256　N
                latent_dim = 12,            # latent dimension 512 D

                cross_heads = 1,             # number of heads for cross attention. paper said 1
                latent_heads = 8,            # number of heads for latent self attention, 8

                cross_dim_head = 12,         # number of dimensions per cross attention head 64
                latent_dim_head = 12,        # number of dimensions per latent self attention head 64

                                             # 出力
                num_classes = 4,             # output number of classes


                attn_dropout = 0.,
                ff_dropout = 0.,

                weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, 
                                             # but can be turned off if you are fourier encoding the data yourself
                self_per_cross_attn = 2      # number of self attention blocks per cross attention
            )

  def forward( self, x ):
    y = self.Encoder( torch.Tensor( x ) )
    return y

class PerceiverQLNetAgent( nn.Module ) :
  def __init__(self):
    super().__init__()
  
    self.e = 0.1
    self.gamma = 0.9
    self.lr = 0.01
    self.action_list = [0,1,2,3]
    self.action_size = len( self.action_list )
    self.PerceiverQLNet = PerceiverQLNet( )


    self.optimizer = torch.optim.SGD(self.PerceiverQLNet.parameters(), lr=self.lr)


  def get_action( self , state ) :
    if np.random.rand() < self.e :
      return np.random.choice( self.action_list )
    else :
      Qs = self.PerceiverQLNet( state )
      return Qs.data.argmax()


  def update( self, state, action, reward, next_state, done ) :
    with torch.no_grad():
      # 次の状態のQ関数を算出
      done = int(done)
      next_Qs = self.PerceiverQLNet( next_state )
      next_Q = next_Qs.max()
    # 未来の価値算出
    buffer = reward + (1-done) * self.gamma * next_Q

    # 今の価値算出
    Qs = self.PerceiverQLNet( state )
    Q = Qs[0,action]
  
    # 損失計算
    loss = torch.nn.functional.mse_loss( buffer, Q )

    # 学習
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.data

env = GridWorld()
agent = PerceiverQLNetAgent()


trial = 1000
history = []

for i in range( trial ) :
  state = env.reset()
  # 初期状態の空間情報
  state_map = env.embedding( state )

  total_loss = 0
  counter = 0
  done = False

  while not done :
    action = agent.get_action( state_map )
    next_state, reward, done = env.step( action )
    # 空間情報を観測
    next_state_map = env.embedding( next_state )
    # 学習した時のロス
    state_map = env.embedding( state )
  

    loss = agent.update( state_map , action , reward, next_state_map , done )
    

    total_loss += loss
    counter += 1
    state = next_state

  print( 'Ep:', i ,'Loss:', total_loss / counter )
  # 各試行ごとのロス変化
  history.append( total_loss / counter )


plt.plot( history )
plt.savefig('img.pdf')

print( torch.reshape(agent.PerceiverQLNet.Encoder.latents[0],(3,4)) )
print( torch.reshape(agent.PerceiverQLNet.Encoder.latents[1],(3,4)) )
print( torch.reshape(agent.PerceiverQLNet.Encoder.latents[2],(3,4)) )
print( torch.reshape(agent.PerceiverQLNet.Encoder.latents[3],(3,4)) )

print( torch.reshape( torch.argmax(agent.PerceiverQLNet.Encoder.latents , 0 ),(3,4)) )
# 0 : 'up'
# 1 : 'down'
# 2 : 'left'
# 3 : 'right'


