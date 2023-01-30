package org.cis120.snakegame;
/**
 * CIS 120 Game HW
 * (c) University of Pennsylvania
 * 
 * @version 2.1, Apr 2017
 */

// imports necessary libraries for Java swing

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * Game Main class that specifies the frame and widgets of the GUI
 */
public class RunSnakeGame implements Runnable {
    public void run() {
        // NOTE : recall that the 'final' keyword notes immutability even for
        // local variables.

        // Top-level frame in which game components live.
        // Be sure to change "TOP LEVEL FRAME" to the name of your game
        final JFrame frame = new JFrame("SNAKE GAME");
        frame.setLocation(300, 300);

        String instructions = "Hi, welcome to the Snake Game! \n\n This is how you play: \n"
                + " 1. You move using arrow keys. \n 2. Collect the apples to " +
                "increase your score. \n"
                + " 3. Don't run into yourself or game is over. \n 4. Don't run into the wall" +
                " or the game is over.\n"
                + " 5. Stay away from the fire. It kills you! \n"
                + " 6. Stay away from the knife. It decreases your score by one. \n \n"
                + " Remember, your score is the length of your snake body! \n \n"
                + " Some features:\n Click the 'Stop Game' button to end the game \n"
                + " Then, after you started a new game, click the 'Last Game' button to see"
                + " how the last game you stopped ended and that last score! \n \n"
                + "Good Luck!";
        JOptionPane.showMessageDialog(frame, instructions, "Snake Game", JOptionPane.OK_OPTION);

        // Status panel
        final JPanel status_panel = new JPanel();
        frame.add(status_panel, BorderLayout.SOUTH);
        final JLabel status = new JLabel("Running...");
        status_panel.add(status);

        // Main playing area
        final org.cis120.snakegame.GameCourt court = new GameCourt(status);
        frame.add(court, BorderLayout.CENTER);

        // Reset button
        final JPanel control_panel = new JPanel();
        frame.add(control_panel, BorderLayout.NORTH);

        // Note here that when we add an action listener to the reset button, we
        // define it as an anonymous inner class that is an instance of
        // ActionListener with its actionPerformed() method overridden. When the
        // button is pressed, actionPerformed() will be called.
        final JButton reset = new JButton("Reset");
        reset.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                court.reset();
            }
        });
        control_panel.add(reset);

        final JButton saveGame = new JButton("Stop Game");
        saveGame.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                court.readGame();
                court.stopMotion();
            }
        });
        control_panel.add(saveGame);

        final JButton lg = new JButton("See Last Game");
        lg.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                court.lastGame();
            }
        });
        control_panel.add(lg);

        // Put the frame on the screen
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        // Start game
        court.reset();
    }
}