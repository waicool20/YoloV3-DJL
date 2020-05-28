package com.waicool20.djl.util

import ai.djl.modality.cv.Image
import java.awt.Dimension
import java.awt.Toolkit
import java.awt.event.KeyEvent
import java.awt.event.KeyListener
import java.awt.event.WindowStateListener
import java.awt.image.BufferedImage
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel

fun Image.openWindowPreview() {
    val dim = Toolkit.getDefaultToolkit().screenSize
    val frame = JFrame("Image")
    frame.add(JLabel(ImageIcon(wrappedImage as BufferedImage)))
    frame.setSize(width, height)
    frame.setLocation(dim.width / 2 - frame.size.width / 2, dim.height / 2 - frame.size.height / 2)
    frame.isVisible = true
    frame.addKeyListener(object : KeyListener {
        override fun keyTyped(event: KeyEvent) {
            frame.removeKeyListener(this)
            frame.dispose()
        }

        override fun keyPressed(event: KeyEvent) = Unit
        override fun keyReleased(event: KeyEvent) = Unit
    })
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
}