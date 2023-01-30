package org.cis120.snakegame;

/**
 * CIS 120 Game HW
 * (c) University of Pennsylvania
 * 
 * @version 2.1, Apr 2017
 */

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.BufferedReader;
import java.io.File;
import java.util.Iterator;
import java.io.FileWriter;
import java.io.FileReader;

/**
 * GameCourt
 *
 * This class holds the primary game logic for how different objects interact
 * with one another. Take time to understand how the timer interacts with the
 * different methods and how it repaints the GUI on every tick().
 */
@SuppressWarnings("serial")
public class GameCourt extends JPanel {

    // the state of the game logic
    private Snake head;
    private Food food;
    private int score = 0;
    private Background b;
    private Fire fire;
    private Knife knife;

    private boolean playing = false; // whether the game is running
    private JLabel status; // Current status text, i.e. "Running..."

    // Game constants
    public static final int COURT_WIDTH = 500;
    public static final int COURT_HEIGHT = 500;
    public static final int HEAD_VELOCITY = 50;

    private int[][] arr = new int[10][10];

    // Update interval for timer, in milliseconds
    public static final int INTERVAL = 170;

    public GameCourt(JLabel status) {
        // creates border around the court area, JComponent method
        setBorder(BorderFactory.createLineBorder(Color.BLACK));

        // The timer is an object which triggers an action periodically with the
        // given INTERVAL. We register an ActionListener with this timer, whose
        // actionPerformed() method is called each time the timer triggers. We
        // define a helper method called tick() that actually does everything
        // that should be done in a single timestep.
        Timer timer = new Timer(INTERVAL, new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                tick();
            }
        });
        timer.start(); // MAKE SURE TO START THE TIMER!

        // Enable keyboard focus on the court area. When this component has the
        // keyboard focus, key events are handled by its key listener.
        setFocusable(true);

        // This key listener allows the square to move as long as an arrow key
        // is pressed, by changing the square's velocity accordingly. (The tick
        // method below actually moves the square.)
        addKeyListener(new KeyAdapter() {
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_LEFT) {
                    head.setVx(-HEAD_VELOCITY);
                    head.setVy(0);

                } else if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
                    head.setVx(HEAD_VELOCITY);
                    head.setVy(0);

                } else if (e.getKeyCode() == KeyEvent.VK_DOWN) {
                    head.setVy(HEAD_VELOCITY);
                    head.setVx(0);

                } else if (e.getKeyCode() == KeyEvent.VK_UP) {
                    head.setVy(-HEAD_VELOCITY);
                    head.setVx(0);
                }
            }
        });

        this.status = status;
    }

    /**
     * (Re-)set the game to its initial state.
     */
    public void reset() {
        food = new Food(COURT_WIDTH, COURT_HEIGHT);
        b = new Background(COURT_WIDTH, COURT_HEIGHT);
        fire = new Fire(COURT_WIDTH, COURT_HEIGHT);
        knife = new Knife(COURT_WIDTH, COURT_HEIGHT);
        head = new Snake(COURT_WIDTH, COURT_HEIGHT);

        score = 0;
        playing = true;
        status.setText("Current Score: " + score);
        arr = new int [10][10];
        // 1 represents place on board with snake
        arr[0][0] = 1;

        // Make sure that this component has the keyboard focus
        requestFocusInWindow();
    }

    public void readGame() {
        try {
            FileWriter f = new FileWriter("position.txt");

            // location of food
            f.write(String.valueOf(food.getPx()) + "\n");
            f.write(String.valueOf(food.getPy()) + "\n");

            // location of background
            f.write(String.valueOf(b.getPx()) + "\n");
            f.write(String.valueOf(b.getPy()) + "\n");

            // location of fire
            f.write(String.valueOf(fire.getPx()) + "\n");
            f.write(String.valueOf(fire.getPy()) + "\n");

            // location of knife
            f.write(String.valueOf(knife.getPx()) + "\n");
            f.write(String.valueOf(knife.getPy()) + "\n");

            // score
            f.write(score + "\n");

            // position of snake
            Iterator i = head.makeIter();
            SnakeNode tail = new SnakeNode(COURT_WIDTH, COURT_HEIGHT);
            tail = (SnakeNode) i.next();

            while (i.hasNext()) {
                SnakeNode temp = new SnakeNode(COURT_WIDTH, COURT_HEIGHT);
                temp = tail;
                tail = (SnakeNode) i.next();
                int a = temp.getPx();
                int b = temp.getPy();

                f.write(String.valueOf(a) + "\n");
                f.write(String.valueOf(b) + "\n");
            }
            f.write(String.valueOf(tail.getPx()) + "\n");
            f.write(String.valueOf(tail.getPy()) + "\n");
            f.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void lastGame() {
        try {
            File f = new File("position.txt");
            FileReader fr = new FileReader(f);
            BufferedReader br = new BufferedReader(fr);

            food.setPx(Integer.parseInt(br.readLine()));
            food.setPy(Integer.parseInt(br.readLine()));

            b.setPx(Integer.parseInt(br.readLine()));
            b.setPy(Integer.parseInt(br.readLine()));

            fire.setPx(Integer.parseInt(br.readLine()));
            fire.setPy(Integer.parseInt(br.readLine()));

            knife.setPx(Integer.parseInt(br.readLine()));
            knife.setPy(Integer.parseInt(br.readLine()));

            int e = Integer.parseInt(br.readLine());
            score = e;

            for (int i = 0; i < score; i++) {
                SnakeNode sn = new SnakeNode(COURT_WIDTH, COURT_HEIGHT);
                sn.setPx(Integer.parseInt(br.readLine()));
                sn.setPy(Integer.parseInt(br.readLine()));
                head.addBeg(sn);
            }
            head.setPx(Integer.parseInt(br.readLine()));
            head.setPy(Integer.parseInt(br.readLine()));
            br.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public boolean allFull(int[][] arr) {
        boolean result = true;
        for (int i = 0; i < arr.length; i ++) {
            for (int j = 0; j < arr[0].length; j++) {
                if (arr[i][j] != 0) {
                    result = false;
                }
            }
        }
        return result;
    }

    public void updateArr(int[][] arr) {
        // place 1 in array where snake body is
        Iterator<SnakeNode> i = head.makeIter();
        SnakeNode tail = i.next();
        while (i.hasNext()) {
            SnakeNode temp = tail;
            tail = i.next();
            int c = temp.getPx() / 50;
            int d = temp.getPy() / 50;
            arr[d][c] = 1;
        }

        // place 1 in array where snake head is
        int e = tail.getPx() / 50;
        int f = tail.getPy() / 50;
        arr[f][e] = 1;

        // place 2 in array where apple is
        int a = food.getPx() / 50;
        int b = food.getPy() / 50;
        arr[b][a] = 2;

        // place 3 in array where fire is
        int g = fire.getPx() / 50;
        int h = fire.getPy() / 50;
        arr[h][g] = 3;

        // place 4 in array where knife is
        int k = knife.getPx() / 50;
        int j = knife.getPy() / 50;
        arr[j][k] = 4;
    }

    public void reset(int [][] arr) {
        for (int i = 0; i < arr.length; i ++) {
            for (int j = 0; j < arr[0].length; j++) {
                arr[i][j] = 0;
            }
        }
    }

    /**
     * This method is called every time the timer defined in the constructor
     * triggers.
     */
    void tick() {
        if (playing) {
            head.move();
            updateArr(arr);
            if (head.hitWall() == Direction.UP
                && head.getVy() == -HEAD_VELOCITY) {
                playing = false;
                status.setText("You lose! Your score was " + score);
            }
            if (head.hitWall() == Direction.DOWN
                && head.getVy() == HEAD_VELOCITY) {
                playing = false;
                status.setText("You lose! Your score was " + score);
            }
            if (head.hitWall() == Direction.LEFT
                && head.getVx() == -HEAD_VELOCITY) {
                playing = false;
                status.setText("You lose! Your score was " + score);
            }
            if (head.hitWall() == Direction.RIGHT
                && head.getVx() == HEAD_VELOCITY) {
                playing = false;
                status.setText("You lose! Your score was " + score);
            }
            if (head.intersects(food)) {
                score++;
                status.setText("Current Score: " + score);
                head.addNode();
            }
            if (head.intersectsWholeBody()) {
                playing = false;
                status.setText("You lose! Your score was " + score);
            }
            if (allFull(arr)) {
                playing = false;
                status.setText("You beat the game! Your score was " + score);
            }

            if (head.intersects(fire)) {
                playing = false;
                status.setText("You lose!");
            }

            if (head.intersects(knife)) {
                head.removeFirst();
                score--;
                status.setText("Current Score: " + score);
            }

            // update the display
            reset(arr);
            repaint();
        }
    }

    public void stopMotion() {
        playing = false;
        status.setText("You lose! Your score was " + score);
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        b.draw(g);
        head.draw(g);

        if (head.getVy() == HEAD_VELOCITY) {
            head.drawHD(g);
        } else if (head.getVy() == -HEAD_VELOCITY) {
            head.drawHU(g);
        } else if (head.getVx() == -HEAD_VELOCITY) {
            head.drawHL(g);
        } else if (head.getVx() == HEAD_VELOCITY) {
            head.drawHR(g);
        } else {
            head.drawHR(g);
        }
        // when the snake hits the apple
        if (head.intersects(food)) {
            int c = food.xVal();
            int d = food.yVal();

            while (arr[d / 50][c / 50] != 0) {
                c = food.xVal();
                d = food.yVal();
            }
            food.updatePosition(c, d);
        }
        if (head.intersects(fire)) {
            int c = fire.xVal();
            int d = fire.yVal();

            while (arr[d / 50][c / 50] != 0) {
                c = fire.xVal();
                d = fire.yVal();
            }
            fire.updatePosition(c, d);
        }
        if (head.intersects(knife)) {
            int c = knife.xVal();
            int d = knife.yVal();

            while (arr[d / 50][c / 50] != 0) {
                c = knife.xVal();
                d = knife.yVal();
            }
            knife.updatePosition(c, d);
        }

        fire.draw(g);
        knife.draw(g);
        food.draw(g);
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(COURT_WIDTH, COURT_HEIGHT);
    }
}