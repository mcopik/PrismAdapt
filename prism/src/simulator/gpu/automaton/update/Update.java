//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.automaton.update;

import java.util.ArrayList;
import java.util.List;

import parser.ast.Expression;
import parser.ast.ExpressionVar;
import parser.ast.Updates;
import prism.Pair;
import simulator.gpu.automaton.AbstractAutomaton.StateVector;
import simulator.gpu.automaton.PrismVariable;

public class Update
{
	private List<Pair<Rate, Action>> updatesList = new ArrayList<>();

	public Update(Updates updates, StateVector variables)
	{
		int updatesCount = updates.getNumUpdates();
		for (int i = 0; i < updatesCount; ++i) {
			parser.ast.Update cur = updates.getUpdate(i);
			Expression prob = updates.getProbability(i);
			Rate rate = new Rate();
			Action action = new Action();
			rate.addRate(prob);
			int exprCount = cur.getNumElements();
			for (int j = 0; j < exprCount; ++j) {
				action.addExpr(variables.getVar(cur.getVar(j)), cur.getExpression(j));
			}
			updatesList.add(new Pair<>(rate, action));
		}
	}

	public int getActionsNumber()
	{
		return updatesList.size();
	}

	public Rate getRate(int updateNumber)
	{
		return updatesList.get(updateNumber).first;
	}

	public Action getAction(int updateNumber)
	{
		return updatesList.get(updateNumber).second;
	}

	/**
	 * Find actions labeled "true" or e.g. "q=q"
	 * @param updateNumber
	 * @return true if action does not change state vector
	 */
	public boolean isActionTrue(int updateNumber)
	{
		Action action = updatesList.get(updateNumber).second;
		if (action.expressions.size() == 0) {
			return true;
		} else if (action.expressions.size() == 1) {
			Expression expr = action.expressions.get(0).second;
			PrismVariable var = action.expressions.get(0).first;
			if (expr instanceof ExpressionVar) {
				ExpressionVar varExpr = (ExpressionVar) expr;
				if (varExpr.getName().equals(var.name)) {
					return true;
				}
			}
		}
		return false;
	}

	public String toString()
	{
		StringBuilder builder = new StringBuilder();
		for (Pair<Rate, Action> pair : updatesList) {
			builder.append(pair.first).append(":").append(pair.second).append(" ");
		}
		return builder.toString();
	}
}