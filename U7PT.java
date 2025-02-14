private static int partTwo(ArrayList<Brick> bricks) {
    // Find the farthest end position so we know how big to make our height array
    int maxEnd = 0;
    for (Brick brick : bricks) {
        if (brick.getEnd() > maxEnd) {
            maxEnd = brick.getEnd();
        }
    }
    
    // Create an ArrayList for heights, initialized to 0, size = maxEnd + 1 
    // (so we can index directly by position)
    ArrayList<Integer> heights = new ArrayList<>();
    for (int i = 0; i <= maxEnd; i++) {
        heights.add(0);
    }
    
    int maxHeight = 0; // track the overall tallest point

    // Now place each brick
    for (Brick brick : bricks) {
        int start = brick.getStart();
        int end = brick.getEnd();
        
        // Find the highest existing stack under this brick’s range
        int currentMax = 0;
        for (int pos = start; pos <= end; pos++) {
            currentMax = Math.max(currentMax, heights.get(pos));
        }
        
        // The new brick goes one level above whatever was highest
        int newBrickHeight = currentMax + 1;
        
        // Update that entire [start..end] range to this new height
        for (int pos = start; pos <= end; pos++) {
            heights.set(pos, newBrickHeight);
        }
        
        // Keep track of the global max
        maxHeight = Math.max(maxHeight, newBrickHeight);
    }
    
    return maxHeight;
}