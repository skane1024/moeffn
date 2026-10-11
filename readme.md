



在大模型MOE的推理中，token的数据是(512, 1024)，其中，512是token的个数，1024是每个 token的特征维度，会把它转换成NCHW_NP128的格式, 数据的shape就是(1, 8,512,128) ，把每个 token 的 1024 维按 128 一组，均匀切成了 8 ， 物理内存上存放的时候，并不是按 token 顺序存的，而是按“组”来存的。先把所有 512 个 token 的第一组特征（也就是每个 token 的前 128 维）连续地排好，然后再排所有 token 的第二组特征（128~255 维），就这样一组一组往后排，直到排完 8 组，这样如果想取出某一个完整 token 的数据，就没法一次性连续读完了。因为它的 1024 维被分散到了 8 个不同的数据块里，你必须分 8 次去取，每次从对应的那个块里取出 128 个数，这个 token 完整的特征向量。

在MOE expert的推理中，每个专家会被编译成一个固定shape的NPU子图，比如一次可以处理64个token, 输入输出的数据格式也是NCHW_NP128，专家是串行的，每次加载一个专家进行计算， 会先找到当前专家下的所有token, 按照NCHW_NP128的格式组织在一起，每次从多个位置取一个token的数据，再存在在多个位置上。需要做gather的操作


NPU 输入输出的数据都是在DDR上的，专家计算后，会在VDSP上token该专家上计算加权求和的结果，  加权求和的结果放在buffer A上  输入数据放在buffer B上  第一次先取buffer B上的数据，乘以权重做加权，将结果放到 buffer A上，  第二次就是再取buffer B上的新数据，乘以权重做加权， 再取buffer A上的数据和加权的结果做求和
VDSP上的计算需要把数据搬运到VDSP的local_memory上， local_memory  容量很小，只有64kb

// 总数据	512 tokens × 1024 维（存储在 DDR）
// 专家数 116
// 每专家 token 数	大约64（非连续，需 gather）gather到npu_input DDR上, 送入NPU计算一次，得到同样shape的结果，放在npu_output DDR上
// local_memory 容量有限 64kb, 	每次 10 个 token（10 × 1024）
// 计算目标	对每和个专家内 64个 token 特征做加权乘法，每个 token的权重不同，再对 116 个专家下每个token的结果求和
// 约束	专家间串行（处理完一个专家再处理下一个）
// Gather 的本质：通过索引表 expert_token_idx 将 DDR 上非连续的 token 地址逐一映射到 local_memory 的连续空间中。
// partial_sum  local memory复用：每个专家处理前清零，处理完后累加到 global_sum，然后清零给下一个专家复用


数据gather的问题这里先不考虑 , 就认为有一个函数处理数据的搬运
这个设计是否正确
是否有优化的空间
补充一些数据搬运的细节  这个伪代码不好理解，整理成更方便理解的伪代码


Local Memory 布局优化
当前使用了 data_buf、partial_sum、readback_buf 三个主要 buffer，各 16 KB，共 48 KB。实际上可以合并为一个 buffer 对：
• 将专家输出读到 buf_A（8×1024），乘上权重后原地更新为加权结果。
• 另用一个 buf_B 读回 global_sum 对应位置。
• 计算 buf_B += buf_A，然后将 buf_B 写回 global_sum。
这样只需两个 16 KB buffer，节省 16 KB，可用于增大 batch size（例如将每批提升到 10 token，进一步减少 DMA 次数）。


//这里NPU计算和VDSP计算是会流水线并行的，目前先串行，梳理清楚整个流程


对齐与 burst：每次 128 维 = 256B（fp16），恰好是常见的 DDR burst 友好粒度；DMA 描述符应按 128 维对齐组织，避免跨 burst 的残缺传输。
//硬件是否支持stride的访问搬运？
写回的必要性判断：如果某 token 在后续专家中还会出现，则写回 global_sum 是必须的；如果硬件支持 DDR 端的原子加（或者 VDSP 有 write-back-with-accumulate 模式），可以省掉 "读回旧值" 这一路，流量从 6KB/token 降到 4KB/token，值得查一下芯片手册。




## token并行
由于专家权重是切分在不同 Cluster 上的（Tensor 并行），每个 Cluster 上的 NPU 计算出的结果是部分和（Partial Sum）。这些部分和通过 reduce_sum 汇聚到 VDSP 的特定内存区域。如果 Token 在多个专家下有结果，VDSP 需要将属于同一个 Token 的不同专家输出从内存中读取出来进行加权合并。

构建全局共享的“输出缓冲区”​在单 Cluster 的内存中，开辟一块连续的区域作为全局输出缓冲区（Global Output Buffer）​，模拟 4 个 Cluster 聚合后的状态。
数据结构设计：每个 Token 对应的结果区域应包含三部分：

累加值（Accumulator）​：存储当前的加权和结果（初始为 0）。 完成计数器（Counter）​：记录该 Token 已经累加了多少个专家的结果（初始为 0）。 有效标志位（Valid Flag）​：可选，用于快速判断该位置是否被写过。


输入阶段（Token 并行）​：Token 被切分分布在 4 个 Cluster 上，每个 Cluster 持有不同的 Token 子集（数量不等）。 计算阶段（专家并行/Tensor 并行）​：专家权重被切分，但计算时需要聚合该专家负责的所有 Token。



