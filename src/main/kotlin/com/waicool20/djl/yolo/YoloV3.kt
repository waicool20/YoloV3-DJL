package com.waicool20.djl.yolo

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.*
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import com.waicool20.djl.util.*

class YoloV3(
    val numClasses: Int = 80,
    val anchors: Array<FloatArray> = arrayOf(
        floatArrayOf(
            116f, 90f,
            156f, 198f,
            373f, 326f
        ),
        floatArrayOf(
            30f, 61f,
            62f, 45f,
            59f, 119f
        ),
        floatArrayOf(
            10f, 13f,
            16f, 30f,
            33f, 23f
        ),
    )
) : AbstractBlock(0) {
    private val skip36Block by lazy {
        SequentialBlock {
            ConvBlock(size = 3, numFilters = 32, stride = 1)
            ConvBlock(size = 3, numFilters = 64, stride = 2)
            ShortcutBlock {
                ConvBlock(size = 1, numFilters = 32, stride = 1, pad = 0)
                ConvBlock(size = 3, numFilters = 64, stride = 1)
            }
            ConvBlock(size = 3, numFilters = 128, stride = 2)
            repeat(2) {
                ShortcutBlock {
                    ConvBlock(size = 1, numFilters = 64, stride = 1, pad = 0)
                    ConvBlock(size = 3, numFilters = 128, stride = 1)
                }
            }
            ConvBlock(size = 3, numFilters = 256, stride = 2)
            repeat(8) {
                ShortcutBlock {
                    ConvBlock(size = 1, numFilters = 128, stride = 1, pad = 0)
                    ConvBlock(size = 3, numFilters = 256, stride = 1)
                }
            }
        }
    }

    private val skip61Block by lazy {
        SequentialBlock {
            ConvBlock(size = 3, numFilters = 512, stride = 2)
            repeat(8) {
                ShortcutBlock {
                    ConvBlock(size = 1, numFilters = 256, stride = 1, pad = 0)
                    ConvBlock(size = 3, numFilters = 512, stride = 1)
                }
            }
        }
    }

    private val yolo82Block by lazy {
        SequentialBlock {
            ConvBlock(size = 3, numFilters = 1024, stride = 2)
            repeat(4) {
                ShortcutBlock {
                    ConvBlock(size = 1, numFilters = 512, stride = 1, pad = 0)
                    ConvBlock(size = 3, numFilters = 1024, stride = 1)
                }
            }
            ConvBlock(size = 1, numFilters = 512, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 1024, stride = 1)
            ConvBlock(size = 1, numFilters = 512, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 1024, stride = 1)
            ConvBlock(size = 1, numFilters = 512, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 1024, stride = 1)
            ConvBlock(
                size = 1,
                numFilters = 3 * (5 + numClasses),
                stride = 1,
                pad = 0,
                batchNorm = false,
                leakyRelu = false
            )
        }
    }

    private val yolo82PreConcatBlock by lazy {
        SequentialBlock {
            ConvBlock(size = 1, numFilters = 256, stride = 1, pad = 0)
            add(LambdaBlock { NDList(it.singletonOrThrow().repeat(longArrayOf(2, 2))) })
        }
    }

    private val yolo94Block by lazy {
        SequentialBlock {
            ConvBlock(size = 1, numFilters = 256, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 512, stride = 1)
            ConvBlock(size = 1, numFilters = 256, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 512, stride = 1)
            ConvBlock(size = 1, numFilters = 256, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 512, stride = 1)
            ConvBlock(
                size = 1,
                numFilters = 3 * (5 + numClasses),
                stride = 1,
                pad = 0,
                batchNorm = false,
                leakyRelu = false
            )
        }
    }

    private val yolo94PreConcatBlock by lazy {
        SequentialBlock {
            ConvBlock(size = 1, numFilters = 128, stride = 1, pad = 0)
            add(LambdaBlock { NDList(it.singletonOrThrow().repeat(longArrayOf(2, 2))) })
        }
    }

    private val yolo106Block by lazy {
        SequentialBlock {
            ConvBlock(size = 1, numFilters = 128, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 256, stride = 1)
            ConvBlock(size = 1, numFilters = 128, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 256, stride = 1)
            ConvBlock(size = 1, numFilters = 128, stride = 1, pad = 0)
            ConvBlock(size = 3, numFilters = 256, stride = 1)
            ConvBlock(size = 1, numFilters = 3 * (5 + numClasses), stride = 1, pad = 0)
        }
    }

    init {
        addChildBlock("Skip36Block", skip36Block)
        addChildBlock("Skip61Block", skip61Block)
        addChildBlock("Yolo82Block", yolo82Block)
        addChildBlock("Yolo82PreConcatBlock", yolo82PreConcatBlock)
        addChildBlock("Yolo94Block", yolo94Block)
        addChildBlock("Yolo94PreConcatBlock", yolo94PreConcatBlock)
        addChildBlock("Yolo106Block", yolo106Block)
    }

    private lateinit var manager: NDManager

    override fun forward(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?
    ): NDList {
        var x = inputs
        x = skip36Block.forward(parameterStore, x, training, params)
        val skip36 = x

        x = skip61Block.forward(parameterStore, x, training, params)
        val skip61 = x

        x = yolo82Block.forward(parameterStore, x, training, params)
        val yolo82 = x

        x = yolo82PreConcatBlock.forward(parameterStore, x, training, params)
        x = NDList(x.singletonOrThrow().concat(skip61.singletonOrThrow(), 1))
        x = yolo94Block.forward(parameterStore, x, training, params)
        val yolo94 = x

        x = yolo94PreConcatBlock.forward(parameterStore, x, training, params)
        x = NDList(x.singletonOrThrow().concat(skip36.singletonOrThrow(), 1))
        x = yolo106Block.forward(parameterStore, x, training, params)
        val yolo106 = x

        manager = inputs[0].manager.newSubManager(inputs[0].device)
        val inputShapesArray = manager.create(inputShapes[0].shape)
        val anchorsArray = manager.create(anchors).div(inputShapes[0].shape[2])

        return NDList(
            inputShapesArray,
            anchorsArray,
            transformOutput(yolo82.singletonOrThrow(), anchorsArray[0]),
            transformOutput(yolo94.singletonOrThrow(), anchorsArray[1]),
            transformOutput(yolo106.singletonOrThrow(), anchorsArray[2])
        )
    }

    private fun transformOutput(output: NDArray, anchors: NDArray): NDArray {
        // (bs, 3 * (5 + numClasses), size, size)
        var array = output
        // (bs, size, size, 3 * (5 + numClasses))
        array = array.transpose(0, 2, 3, 1)
        // (bs, size, size, 3, 5 + numClasses)
        array = array.reshape(array.shape[0], array.shape[1], array.shape[2], 3, 5L + numClasses)
        // (bs, 3, size, size, 5 + numClasses)
        array = array.transpose(0, 3, 1, 2, 4)

        val gridSize = array.shape[2]

        /*
            Creates an nd array with y offsets, size depends on the current grid size. eg. 3x3
            0 0 0
            1 1 1
            2 2 2
         */
        val gridY = manager.arange(gridSize.toFloat()).repeat(gridSize).reshape(1, 1, gridSize, gridSize)

        /*
            Creates an nd array with x offsets, size depends on the current grid size. eg. 3x3
            0 1 2
            0 1 2
            0 1 2
        */
        val gridX = gridY.duplicate().transpose(0, 1, 3, 2)

        val anchorsArray = anchors.expandDims(0).reshape(3, 2)
        val anchorW = anchorsArray.get(NDIndex(":, 0:1")).reshape(1, 3, 1, 1)
        val anchorH = anchorsArray.get(NDIndex(":, 1:2")).reshape(1, 3, 1, 1)

        // Transform xy coordinates that are relative to the grid cell to absolute image coordinates
        // New xy will be the center coordinates of the resulting bounding box
        val x = array.get(NDIndex(":, :, :, :, 0")).ndArrayInternal.sigmoid().add(gridX).div(gridSize).expandDims(4)
        val y = array.get(NDIndex(":, :, :, :, 1")).ndArrayInternal.sigmoid().add(gridY).div(gridSize).expandDims(4)

        // Transform wh coordinates according to the pre-defined anchors
        // New wh will be full wh of the resulting bounding box
        val w = array.get(NDIndex(":, :, :, :, 2")).exp().mul(anchorW).expandDims(4)
        val h = array.get(NDIndex(":, :, :, :, 3")).exp().mul(anchorH).expandDims(4)
        val p = array.get(NDIndex(":, :, :, :, 4")).ndArrayInternal.sigmoid().expandDims(4)
        val c = array.get(NDIndex(":, :, :, :, 5:")).ndArrayInternal.sigmoid()

        return NDArrays.concat(NDList(x, y, w, h, p, c), 4)
    }

    override fun getOutputShapes(manager: NDManager, inputShapes: Array<out Shape>): Array<Shape> {
        // TODO
        return arrayOf(Shape(256, 256))
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        var shape = inputShapes
        val skip36Shape = skip36Block.initialize(manager, dataType, *shape)
        val skip61Shape = skip61Block.initialize(manager, dataType, *skip36Shape)
        shape = yolo82Block.initialize(manager, dataType, *skip61Shape)
        shape = yolo82PreConcatBlock.initialize(manager, dataType, *shape)
        shape = yolo94Block.initialize(
            manager,
            dataType,
            Shape(shape[0][0], shape[0][1] + skip61Shape[0][1], shape[0][2], shape[0][3])
        )
        shape = yolo94PreConcatBlock.initialize(manager, dataType, *shape)
        yolo106Block.initialize(
            manager,
            dataType,
            Shape(shape[0][0], shape[0][1] + skip36Shape[0][1], shape[0][2], shape[0][3])
        )
    }

    private fun SequentialBlock.ConvBlock(
        size: Long,
        numFilters: Int,
        stride: Long,
        pad: Long = 1,
        batchNorm: Boolean = true,
        leakyRelu: Boolean = true
    ): SequentialBlock {
        Conv2D(
            kernel = Shape(size, size),
            numFilters = numFilters,
            stride = Shape(stride, stride),
            pad = Shape(pad, pad)
        )
        if (batchNorm) BatchNorm(epsilon = 0.001f)
        if (leakyRelu) LeakyRelu(alpha = 0.1f)
        return this
    }
}

