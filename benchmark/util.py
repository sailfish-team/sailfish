def summarize(timing_infos, blocks):
    mlups_total = 0
    mlups_comp = 0
    send_time = 0.0
    recv_time = 0.0
    for ti in timing_infos:
        block = blocks[ti.block_id]
        mlups_total += block.num_nodes / ti.total * 1e-6
        mlups_comp  += block.num_nodes / ti.comp * 1e-6
        send_time += ti.send
        recv_time += ti.recv

    n = len(blocks)
    return list(blocks[0].size) + [block.num_nodes, mlups_total, mlups_comp, send_time / n, recv_time / n]
