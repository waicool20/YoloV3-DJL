package com.waicool20.djl.util

import ai.djl.ndarray.NDList
import ai.djl.nn.Block
import ai.djl.nn.Blocks
import ai.djl.nn.ParallelBlock
import ai.djl.nn.SequentialBlock

class ShortcutBlock(block: Block) : ParallelBlock({
    val unit = it[0].singletonOrThrow()
    val parallel = it[1].singletonOrThrow()
    NDList(unit.add(parallel).ndArrayInternal.relu())
}, listOf(block, Blocks.identityBlock())) {
    override fun toString(): String {
        return super.toString().replaceFirst("Parallel", "Shortcut")
    }
}

fun ShortcutBlock(action: SequentialBlock.() -> Unit): Block {
    return ShortcutBlock(SequentialBlock().apply(action))
}