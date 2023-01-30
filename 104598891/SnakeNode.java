package org.cis120.snakegame;

/**
 * CIS 120 Game HW
 * (c) University of Pennsylvania
 * 
 * @version 2.1, Apr 2017
 */

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * A basic game object starting in the upper left corner of the game court. It
 * is displayed as a square of a specified color.
 */
public class SnakeNode extends GameObj {
    public static final String IMG_FILE = "files/snake.png";
    public static final String HEAD_LEFT = "files/snakeL.png";
    public static final String HEAD_RIGHT = "files/snakeR.png";
    public static final String HEAD_DOWN = "files/snakeD.png";
    public static final String HEAD_UP = "files/snakeU.png";

    public static final int SIZE = 48;
    public static final int INIT_POS_X = 1;
    public static final int INIT_POS_Y = 1;
    public static final int INIT_VEL_X = 0;
    public static final int INIT_VEL_Y = 0;

    private static BufferedImage img;
    private static BufferedImage h1;
    private static BufferedImage h2;
    private static BufferedImage h3;
    private static BufferedImage h4;

    /**
     * Note that, because we don't need to do anything special when constructing
     * a Square, we simply use the superclass constructor called with the
     * correct parameters.
     */
    public SnakeNode(int courtWidth, int courtHeight) {
        super(INIT_VEL_X, INIT_VEL_Y, INIT_POS_X, INIT_POS_Y, SIZE, SIZE, courtWidth, courtHeight);
        try {
            if (img == null) {
                img = ImageIO.read(new File(IMG_FILE));
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }

        try {
            if (h1 == null) {
                h1 = ImageIO.read(new File(HEAD_LEFT));
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }

        try {
            if (h2 == null) {
                h2 = ImageIO.read(new File(HEAD_RIGHT));
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }

        try {
            if (h3 == null) {
                h3 = ImageIO.read(new File(HEAD_DOWN));
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }

        try {
            if (h4 == null) {
                h4 = ImageIO.read(new File(HEAD_UP));
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }
    }

    @Override
    public void draw(Graphics g) {
        g.drawImage(img, this.getPx(), this.getPy(), this.getWidth(), this.getHeight(), null);
    }

    public void drawHL(Graphics g) {
        g.drawImage(h1, this.getPx(), this.getPy(), this.getWidth(), this.getHeight(), null);
    }
    public void drawHR(Graphics g) {
        g.drawImage(h2, this.getPx(), this.getPy(), this.getWidth(), this.getHeight(), null);
    }
    public void drawHD(Graphics g) {
        g.drawImage(h3, this.getPx(), this.getPy(), this.getWidth(), this.getHeight(), null);
    }
    public void drawHU(Graphics g) {
        g.drawImage(h4, this.getPx(), this.getPy(), this.getWidth(), this.getHeight(), null);
    }
}