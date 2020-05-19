package com.waicool20.djl.yolo

import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.loss.AbstractCompositeLoss
import ai.djl.training.loss.Loss
import ai.djl.util.Pair
import com.waicool20.djl.util.YoloUtils.bboxIOU
import com.waicool20.djl.util.YoloUtils.whIOU
import com.waicool20.djl.util.times
import kotlin.math.floor
import kotlin.math.roundToLong

class YoloV3Loss(
    val ignoreThreshold: Double = 0.5
) : AbstractCompositeLoss("YoloV3Loss") {

    private inner class YoloV3LayerLoss(private val layerIndex: Int) : AbstractCompositeLoss("YoloV3LayerLoss") {
        init {
            components = listOf(
                Loss.l2Loss("BoundingBoxXLoss"),
                Loss.l2Loss("BoundingBoxYLoss"),
                Loss.l2Loss("BoundingBoxWLoss"),
                Loss.l2Loss("BoundingBoxHLoss"),
                Loss.sigmoidBinaryCrossEntropyLoss("ConfObjLoss"),
                Loss.sigmoidBinaryCrossEntropyLoss("ConfNoObjLoss", 100f, false),
                Loss.sigmoidBinaryCrossEntropyLoss("ClassLoss")
            )
        }

        override fun inputForComponent(componentIndex: Int, labels: NDList, predictions: NDList): Pair<NDList, NDList> {
            val targets = buildTargets(labels, predictions)
            val iouScores = targets[0]
            val classMask = targets[1]
            val objMask = targets[2]
            val noObjMask = targets[3]
            val tx = targets[4]
            val ty = targets[5]
            val tw = targets[6]
            val th = targets[7]
            val tcls = targets[8]
            val tconf = targets[9]
            val output = predictions[layerIndex + 2]
            return when (componentIndex) {
                0 -> {
                    val x = output.get(NDIndex(":, :, :, :, 0")).reshape(objMask.shape) * objMask
                    Pair(NDList(tx), NDList(x))
                }
                1 -> {
                    val y = output.get(NDIndex(":, :, :, :, 1")).reshape(objMask.shape) * objMask
                    Pair(NDList(ty), NDList(y))
                }
                2 -> {
                    val w = output.get(NDIndex(":, :, :, :, 2")).reshape(objMask.shape) * objMask
                    Pair(NDList(tw), NDList(w))
                }
                3 -> {
                    val h = output.get(NDIndex(":, :, :, :, 3")).reshape(objMask.shape) * objMask
                    Pair(NDList(th), NDList(h))
                }
                4 -> {
                    val tObjConf = tconf * objMask
                    val conf = output.get(NDIndex(":, :, :, :, 4")).reshape(objMask.shape) * objMask
                    Pair(NDList(tObjConf), NDList(conf))
                }
                5 -> {
                    val tNoObjConf = tconf * noObjMask
                    val conf = output.get(NDIndex(":, :, :, :, 4")).reshape(noObjMask.shape) * noObjMask
                    Pair(NDList(tNoObjConf), NDList(conf))
                }
                6 -> {
                    val mask = objMask.expandDims(0).repeat(0, tcls.shape[4]).transpose(1, 2, 3, 4, 0)
                    val cls = output.get(NDIndex(":, :, :, :, 5:")).reshape(mask.shape) * mask
                    Pair(NDList(tcls), NDList(cls))
                }
                else -> error("Invalid component index: $componentIndex")
            }
        }

        /**
         * @see <a href="https://towardsdatascience.com/calculating-loss-of-yolo-v3-layer-8878bfaaf1ff">Calculating Loss</a>
         */
        private fun buildTargets(labels: NDList, predictions: NDList): NDList {
            val anchors = predictions[1][layerIndex.toLong()].reshape(3, 2)
            val output = predictions[layerIndex + 2]
            var target = labels.singletonOrThrow()
            target = target.reshape(target.shape[0], 5)

            // Batch size
            val nB = output.shape[0]
            // Anchors
            val nA = output.shape[1]
            // Classes
            val nC = output.shape[4] - 5
            // Stride
            val nG = output.shape[3]

            val maskShape = Shape(nB, nA, nG, nG)

            val objMask = manager.zeros(maskShape)
            val noObjMask = manager.ones(maskShape)
            val classMask = manager.zeros(maskShape)
            val iouScores = manager.zeros(maskShape)
            val tx = manager.zeros(maskShape)
            val ty = manager.zeros(maskShape)
            val tw = manager.zeros(maskShape)
            val th = manager.zeros(maskShape)
            val tcls = manager.zeros(maskShape.addAll(Shape(nC)))

            val targetBoxes = target.get(NDIndex(":, 1:5")).mul(nG)
            val gxy = targetBoxes.get(NDIndex(":, :2"))
            val gwh = targetBoxes.get(NDIndex(":, 2:"))

            val ious = NDArrays.concat(NDList(whIOU(anchors[0], gwh), whIOU(anchors[1], gwh), whIOU(anchors[2], gwh)))
                .reshape(nA, nB)
            //val bestIOU = ious.max(intArrayOf(0))
            val bestIOUn = ious.argMax(0)

            for (b in 0 until nB) {
                val gx = gxy.getFloat(b, 0)
                val gy = gxy.getFloat(b, 1)
                val gi = gx.roundToLong()
                val gj = gy.roundToLong()

                val index = NDIndex(b, bestIOUn.getLong(b), gj, gi)

                objMask.set(index, 1)
                noObjMask.set(index, 0)
                tx.set(index, gx - floor(gx))
                ty.set(index, gy - floor(gy))

                tcls.set(NDIndex(b, bestIOUn.getLong(b), gj, gi, target.getFloat(b, 0).roundToLong()), 1)

                for (a in 0 until ious.shape[0]) {
                    if (ious.getFloat(a, b) > ignoreThreshold) {
                        noObjMask.set(NDIndex(b, a, gj, gi), 0)
                    }
                }
                if (output.get(index).argMax().getLong() == target.getFloat(b, 0).roundToLong()) {
                    classMask.set(index, 1f)
                }
                val bboxIOU = bboxIOU(output[index], target[NDIndex("$b, 1:5")])
                iouScores.set(index, bboxIOU)
            }
            return NDList(
                iouScores,
                classMask,
                objMask,
                noObjMask,
                tx, ty,
                tw, th, tcls, objMask.toType(DataType.FLOAT32, true)
            )
        }
    }

    private lateinit var manager: NDManager

    init {
        components = List(3) { i -> YoloV3LayerLoss(i) }
    }

    override fun inputForComponent(componentIndex: Int, labels: NDList, predictions: NDList): Pair<NDList, NDList> {
        manager = predictions[0].manager.newSubManager()

        return when (componentIndex) {
            0 -> Pair(labels, predictions)
            1 -> Pair(labels, predictions)
            2 -> Pair(labels, predictions)
            else -> error("Invalid component index: $componentIndex")
        }
    }
}