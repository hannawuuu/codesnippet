package org.cis120.snakegame;

import java.awt.*;
import java.util.Iterator;
import java.util.LinkedList;

public class Snake {
    private LinkedList<SnakeNode> snakes;
    private SnakeNode head;
    private int courtWidth;
    private int courtHeight;

    public void setPx(int px) {
        head.setPx(px);
    }

    public void setPy(int py) {
        head.setPy(py);
    }

    public void setVx(int vx) {
        head.setVx(vx);
    }

    public void setVy(int vy) {
        head.setVy(vy);
    }

    public int getVx() {
        return head.getVx();
    }

    public int getVy() {
        return head.getVy();
    }

    public Direction hitWall() {
        return head.hitWall();
    }

    public boolean intersects(GameObj that) {
        return head.intersects(that);
    }

    public void drawHL(Graphics g) {
        head.drawHL(g);
    }
    public void drawHR(Graphics g) {
        head.drawHR(g);
    }
    public void drawHD(Graphics g) {
        head.drawHD(g);
    }
    public void drawHU(Graphics g) {
        head.drawHU(g);
    }

    public Snake(int courtWidth, int courtHeight) {
        this.courtHeight = courtHeight;
        this.courtWidth = courtWidth;
        snakes = new LinkedList<SnakeNode>();
        this.head = new SnakeNode(courtWidth, courtHeight);
        snakes.add(head);
    }

    public void move() {
        Iterator<SnakeNode> i = snakes.iterator();
        SnakeNode tail = i.next();
        while (i.hasNext()) {
            SnakeNode temp = tail;
            tail = i.next();
            temp.setPx(tail.getPx());
            temp.setPy(tail.getPy());
            temp.move();
        }
        head.move();
    }

    public void draw(Graphics g) {
        Iterator<SnakeNode> i = snakes.iterator();
        SnakeNode tail = i.next();
        while (i.hasNext()) {
            SnakeNode temp = tail;
            tail = i.next();
            temp.draw(g);
        }
    }

    public void addNode() {
        SnakeNode tail = new SnakeNode(courtWidth, courtHeight);
        SnakeNode oldTail = snakes.peekFirst();
        tail.setPx(oldTail.getPx() - oldTail.getVx() - oldTail.getVy());
        tail.setPy(oldTail.getPy() - oldTail.getVx() - oldTail.getVy());
        snakes.addFirst(tail);
    }

    public void addBeg(SnakeNode s) {
        snakes.addFirst(s);
    }

    public void addLast(SnakeNode s) {
        snakes.add(s);
    }

    public boolean intersectsWholeBody() {
        boolean result = false;
        Iterator<SnakeNode> i = snakes.iterator();
        SnakeNode tail = i.next();
        while (i.hasNext()) {
            SnakeNode temp = tail;
            tail = i.next();
            if (head.willIntersect(temp)) {
                result = true;
            }
        }
        return result;
    }

    public int getLength() {
        Iterator<SnakeNode> i = snakes.iterator();
        SnakeNode tail = i.next();
        int count = 1;
        while (i.hasNext()) {
            count++;
            tail = i.next();
        }
        return count;
    }

    public void removeFirst() {
        snakes.removeFirst();
    }

    public Iterator makeIter() {
        Iterator<SnakeNode> i = snakes.iterator();
        return i;
    }

}
