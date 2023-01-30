package org.cis120.snakegame;

import org.junit.jupiter.api.Test;

import javax.swing.*;

import static org.junit.jupiter.api.Assertions.*;

public class GameTest {

    @Test
    public void testSnakeAddNode() {
        Snake s = new Snake(500, 500);
        s.addNode();
        s.addNode();
        s.addNode();

        assertEquals(4, s.getLength());
    }

    @Test
    public void testSnakeMove() {
        Snake s = new Snake(500, 500);
        SnakeNode one = new SnakeNode(500, 500);
        one.setPx(50);
        SnakeNode two = new SnakeNode(500, 500);
        two.setPx(50);
        two.setPy(50);
        SnakeNode three = new SnakeNode(500, 500);
        three.setPy(50);

        s.addBeg(one);
        s.addBeg(two);
        s.addBeg(three);

        s.move();

        assertNotEquals(50, one.getPx());
        assertEquals(50, two.getPx());
        assertEquals(50, two.getPx());
    }

    @Test
    public void testSnakeMove2() {
        Snake s = new Snake(500, 500);
        SnakeNode one = new SnakeNode(500, 500);
        one.setPx(50);
        SnakeNode two = new SnakeNode(500, 500);
        two.setPx(50);
        two.setPy(50);
        SnakeNode three = new SnakeNode(500, 500);
        three.setPy(50);

        s.addBeg(one);
        s.addBeg(two);
        s.addBeg(three);

        s.move();

        assertEquals(50, two.getPx());
    }

    @Test
    public void testSnakeMove3() {
        Snake s = new Snake(500, 500);
        SnakeNode one = new SnakeNode(500, 500);
        one.setPx(50);
        SnakeNode two = new SnakeNode(500, 500);
        two.setPx(50);
        two.setPy(50);
        SnakeNode three = new SnakeNode(500, 500);
        three.setPy(50);

        s.addBeg(one);
        s.addBeg(two);
        s.addBeg(three);

        s.move();

        assertEquals(50, two.getPx());
    }

    @Test
    public void testSnakeMove4() {
        Snake s = new Snake(500, 500);
        SnakeNode one = new SnakeNode(500, 500);
        one.setPx(50);
        SnakeNode two = new SnakeNode(500, 500);
        two.setPx(50);
        two.setPy(50);
        SnakeNode three = new SnakeNode(500, 500);
        three.setPy(50);

        s.addBeg(one);
        s.addBeg(two);
        s.addBeg(three);

        s.move();

        assertEquals(1, two.getPy());
    }

    @Test
    public void testSnakeRemoveFirst() {
        Snake s = new Snake(500, 500);
        s.addNode();
        s.addNode();
        s.addNode();
        s.removeFirst();
        assertEquals(3, s.getLength());
    }

    @Test
    public void testArrFull() {
        int[][] arr = new int[5][5];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                arr[i][j] = (int) Math.random() * 10;
            }
        }
    }
}
