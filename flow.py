from jina import Flow
from executor import TextReward


flow = (
    Flow()
    .add(uses=TextReward, timeout_ready=-1,reload=True)
)

with flow:
    flow.block()