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
 * A game object displayed using an image.
 *
 * Note that the image is read from the file when the object is constructed, and
 * that all objects created by this constructor share the same image data (i.e.
 * img is static). This is important for efficiency: your program will go very
 * slowly if you try to create a new BufferedImage every time the draw method is
 * invoked.
 */
public class Food extends GameObj {
    public static final String IMG_FILE = "files/apple.png";
    public static final int SIZE = 50;
    public static final int INIT_VEL_X = 0;
    public static final int INIT_VEL_Y = 0;

    private static BufferedImage img;

    public int xVal() {
        int gridXPos = (int)(Math.random() * 10);
        gridXPos = gridXPos * 50;
        return gridXPos;
    }

    public int yVal() {
        int gridYPos = (int)(Math.random() * 10);
        gridYPos = gridYPos * 50;
        return gridYPos;
    }

    public Food(int courtWidth, int courtHeight) {
        super(INIT_VEL_X, INIT_VEL_Y, 0, 0, SIZE, SIZE, courtWidth, courtHeight);

        this.setPx(xVal());
        this.setPy(yVal());

        try {
            if (img == null) {
                img = ImageIO.read(new File(IMG_FILE));
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }
    }

    public void updatePosition(int x, int y) {
        this.setPx(x);
        this.setPy(y);
    }

    @Override
    public void draw(Graphics g) {
        g.drawImage(img, this.getPx(), this.getPy(), this.getWidth(), this.getHeight(), null);
    }
}
