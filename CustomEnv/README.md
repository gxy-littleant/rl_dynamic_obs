1. 在 /Users/gaoxuanyu/miniconda3/envs/minigrid/lib/python3.9/site-packages/gymnasium/envs 目录中注册自己的环境
    eg:
        register(
        id='CustomDynamicObs-v0', 
        entry_point="gymnasium.envs.classic_control.CustomDynamicObs-v0",
        )

    注意命名要规范，否则可能会报错

2. 将自定义的环境复制到 /Users/gaoxuanyu/miniconda3/envs/minigrid/lib/python3.9/site-packages/gymnasium/envs/classic_control 目录中
.py文件名称应该为在步骤1中注册的id名称