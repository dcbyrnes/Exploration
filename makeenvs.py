for i in range(100):
	print("register(id='small-maze-{}-v0', entry_point='gym_maze.envs:SmallMaze{}', timestep_limit=1000000)".format(i,i))

for i in range(100):
	print("""class SmallMaze{}(MazeEnv):
    def __init__(self):
        super(SmallMaze{}, self).__init__(maze_file='small_mazes/{}.npy', mode='plus')""".format(i,i,i))