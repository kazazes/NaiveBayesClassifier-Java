package com.sibylvision;

public enum NaiveBayesSpamClassification {
	GOOD_CLASSIFICATION(0, 3), BAD_CLASSIFICATION(1, 2), NO_CLASSIFICATION(-1, -1);
	
	private int returnVal;
	private int classificationThreshold;
	
	NaiveBayesSpamClassification(int arrayLocation, int classificationThreshold) {
		this.returnVal = arrayLocation;
		this.classificationThreshold = classificationThreshold;
	}
	
	public int arrayLocation() {
		return returnVal;
	}
	
	public int classificationThreshold() {
		return classificationThreshold;
	}
}
