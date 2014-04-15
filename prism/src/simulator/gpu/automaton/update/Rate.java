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
import prism.PrismLangException;

public class Rate
{
	public double rate = 0.0;
	public List<Expression> expressions = null;
	private boolean isConst = true;

	public Rate()
	{

	}

	public Rate(Rate rate)
	{
		this.rate = rate.rate;
		this.isConst = rate.isConst;
		if (rate.expressions != null) {
			for (Expression expr : rate.expressions) {
				this.expressions = new ArrayList<>();
				this.expressions.add(expr.deepCopy());
			}
		}
	}

	public Rate(double initial)
	{
		rate = initial;
	}

	public void addRate(Expression expr)
	{
		if (expr == null) {
			rate += 1.0;
			return;
		}
		if (expr.isConstant()) {
			add(expr);
		} else {
			expressions = new ArrayList<>();
			expressions.add(expr);
			this.isConst = false;
		}
	}

	public void addRate(Rate rate)
	{
		this.rate += rate.rate;
		if (!rate.isConst) {
			if (isConst) {
				expressions = new ArrayList<>();
			}
			expressions.addAll(rate.expressions);
			this.isConst = false;
		}
	}

	private void add(Expression expr)
	{
		try {
			rate += expr.evaluateDouble();
		} catch (PrismLangException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public boolean isRateConst()
	{
		return isConst;
	}

	public String toString()
	{
		StringBuilder builder = new StringBuilder();
		if (rate != 0.0) {
			builder.append("(").append(rate).append(") +");
		}
		if (expressions != null) {
			for (Expression expr : expressions) {
				builder.append("(").append(expr).append(") +");
			}
		}
		builder.deleteCharAt(builder.length() - 1);
		return builder.toString();
	}
}