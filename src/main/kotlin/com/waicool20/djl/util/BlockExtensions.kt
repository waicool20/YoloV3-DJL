package com.waicool20.djl.util

import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.*
import ai.djl.nn.convolutional.Conv2D
import ai.djl.nn.norm.BatchNorm

fun SequentialBlock(builder: SequentialBlock.() -> Unit): SequentialBlock {
    return SequentialBlock().apply(builder)
}

fun ParallelBlock(vararg blocks: Block, transform: (List<NDList>) -> NDList): ParallelBlock {
    return ParallelBlock(transform, blocks.toList())
}

fun ParallelBlock(blocks: List<Block>, transform: (List<NDList>) -> NDList): ParallelBlock {
    return ParallelBlock(transform, blocks)
}

fun SequentialBlock.Conv2D(
    kernel: Shape,
    numFilters: Int,
    bias: Boolean? = null,
    dilate: Shape? = null,
    numGroups: Int? = null,
    pad: Shape? = null,
    stride: Shape? = null,
    action: Conv2D.() -> Unit = {}
): Conv2D {
    val conv2d = Conv2D.builder().apply {
        setKernel(kernel)
        setNumFilters(numFilters)
        if (bias != null) optBias(bias)
        if (dilate != null) optDilate(dilate)
        if (numGroups != null) optNumGroups(numGroups)
        if (pad != null) optPad(pad)
        if (stride != null) optStride(stride)
    }.build()
    conv2d.action()
    add(conv2d)
    return conv2d
}

fun SequentialBlock.BatchNorm(
    axis: Int? = null,
    center: Boolean? = null,
    epsilon: Float? = null,
    momentum: Float? = null,
    scale: Boolean? = null,
    action: BatchNorm.() -> Unit = {}
): BatchNorm {
    val batchNorm = BatchNorm.builder().apply {
        if (axis != null) optAxis(axis)
        if (center != null) optCenter(center)
        if (epsilon != null) optEpsilon(epsilon)
        if (momentum != null) optMomentum(momentum)
        if (scale != null) optScale(scale)
    }.build()
    batchNorm.action()
    add(batchNorm)
    return batchNorm
}

fun SequentialBlock.LeakyRelu(
    alpha: Float,
    action: LambdaBlock.() -> Unit = {}
): LambdaBlock {
    val leakyRelu = (Activation.leakyReluBlock(alpha) as LambdaBlock).apply(action)
    add(leakyRelu)
    return leakyRelu
}

fun SequentialBlock.ShortcutBlock(
    action: SequentialBlock.() -> Unit
): Block {
    val block = com.waicool20.djl.util.ShortcutBlock(action)
    add(block)
    return block
}