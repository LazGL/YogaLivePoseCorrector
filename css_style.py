css = """
/* Pictogram Container Styling */
#pictogram-container {
    display: flex;
    justify-content: space-evenly; /* Space pictograms evenly */
    align-items: center; /* Align items vertically */
    gap: 20px; /* Space between pictograms */
    margin-top: 20px;
    border: none; /* Remove container border */
    box-shadow: none; /* Remove shadow from container */
}

/* Default Pictogram Styling */
.pictogram img,
.highlighted img {
    width: 200px; /* Fixed size for pictograms */
    height: 200px; /* Fixed size for pictograms */
    border-radius: 8px; /* Rounded corners */
    border: none; /* No border on the images themselves */
    transition: transform 0.3s ease-in-out; /* Smooth zoom effect */
}

/* Highlighted Pictogram Styling */
.highlighted img {
    border: 4px solid blue; /* Add bold blue border to highlighted */
}

/* Hover Effect for Pictograms */
.pictogram img:hover,
.highlighted img:hover {
    transform: scale(1.1); /* Slight zoom effect */
}

/* Responsive Design */
@media (max-width: 768px) {
    #pictogram-container {
        flex-wrap: wrap; /* Wrap pictograms if the screen is too small */
    }
    .pictogram img,
    .highlighted img {
        width: 150px; /* Smaller size for mobile devices */
        height: 150px;
    }
}
"""